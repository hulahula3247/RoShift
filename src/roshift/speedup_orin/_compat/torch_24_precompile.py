# PyTorch CachingAutotuner._precompile_config, verbatim method excerpt.
# Copyright and BSD-3-Clause terms: third_party/pytorch_LICENSE.txt.
# Upstream: https://github.com/pytorch/pytorch/blob/3bcc3cddb5/torch/_inductor/runtime/triton_heuristics.py
# Complete upstream file SHA256:
# 757637fee94802571f531a4c52587dbc00b0961ed33fd7054ea62c7e2faa596e
# The adapter compiles this function in the installed module's globals.
# This source file is not imported directly (its globals belong to PyTorch).
class CachingAutotuner:
    def _precompile_config(self, cfg: Config, warm_cache_only: bool):
        """Ahead of time compile a given autotuner config."""
        compile_meta = copy.deepcopy(self.triton_meta)
        for k, v in cfg.kwargs.items():
            if self.device_props.type != "hip":
                if k == "matrix_instr_nonkdim":
                    compile_meta["matrix_instr_nonkdim"] = v
                    continue
                if k == "waves_per_eu":
                    compile_meta["waves_per_eu"] = v
                    continue
            compile_meta["constants"][self.fn.arg_names.index(k)] = v
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages
        compile_meta["debug"] = self.inductor_meta.get(
            "assert_indirect_indexing", True
        ) and not self.inductor_meta.get("is_hip", False)

        # device type will be "hip" rather than "cuda" here
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        if ASTSource:
            compile_args = (
                ASTSource(
                    self.fn,
                    compile_meta["signature"],
                    compile_meta["constants"],
                    compile_meta["configs"][0],
                ),
            )

            cc_str = str(compile_meta["cc"])
            if "gfx10" in cc_str or "gfx11" in cc_str:
                rocm_warp_size = 32
            else:
                rocm_warp_size = 64

            if GPUTarget:
                target = GPUTarget(
                    compile_meta["device_type"],
                    compile_meta["cc"],
                    rocm_warp_size if torch.version.hip else 32,
                )
            else:
                target = (
                    (compile_meta["device_type"], compile_meta["cc"])
                    if not torch.version.hip
                    else [
                        compile_meta["device_type"],
                        compile_meta["cc"],
                        rocm_warp_size,
                    ]
                )

            options = {
                "num_warps": compile_meta["num_warps"],
                "num_stages": compile_meta["num_stages"],
                "debug": compile_meta["debug"],
            }
            if self.device_props.type != "hip":
                if "waves_per_eu" in compile_meta:
                    options["waves_per_eu"] = compile_meta["waves_per_eu"]
                if "matrix_instr_nonkdim" in compile_meta:
                    options["matrix_instr_nonkdim"] = compile_meta[
                        "matrix_instr_nonkdim"
                    ]
            compile_kwargs = {
                "target": target,
                "options": options,
            }
        else:
            compile_args = (self.fn,)
            compile_kwargs = compile_meta

        if warm_cache_only:
            return (
                triton.compile(*compile_args, **compile_kwargs),
                None,
            )

        # importing from torch is safe now that precompile has returned
        from torch._dynamo.device_interface import DeviceGuard

        device_interface = self.get_device_interface()

        # load binary to the correct device
        with DeviceGuard(device_interface, compile_meta["device"]):  # type: ignore[attr-defined]
            # need to initialize context
            device_interface.synchronize(device_interface.current_device())

            try:
                binary = triton.compile(*compile_args, **compile_kwargs)
            except Exception:
                log.exception(
                    "Triton compilation failed: %s\n%s\nmetadata: %s",
                    self.inductor_meta.get("kernel_name", "triton_"),
                    self.fn.src,
                    compile_meta,
                )
                raise
            binary._init_handles()

        call_args = [
            arg
            for i, arg in enumerate(self.fn.arg_names)
            if i not in self.fn.constexprs
        ]
        def_args = [name for name in self.fn.arg_names if name not in cfg.kwargs]

        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "launch_enter_hook": binary.launch_enter_hook,
            "launch_exit_hook": binary.launch_exit_hook,
            "metadata": binary.packed_metadata
            if hasattr(binary, "packed_metadata")
            else binary.metadata,
            "shared": binary_shared,
        }

        scope["num_warps"] = (
            binary.num_warps
            if hasattr(binary, "num_warps")
            else binary.metadata.num_warps
        )

        scope["cta_args"] = (
            (binary.num_ctas, *get_first_attr(binary, "cluster_dims", "clusterDims"))
            if hasattr(binary, "num_ctas")
            else (
                (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                if hasattr(binary, "metadata")
                else ()
            )
        )

        scope["function"] = get_first_attr(binary, "function", "cu_function")

        def get_launch_args_without_kernel_launch_metadata(
            grid,
            grid_0,
            grid_1,
            grid_2,
            stream,
            function,
            metadata,
            bin,
            launch_enter_hook,
            launch_exit_hook,
            num_warps,
            shared,
            cta_args,
            args,
        ):
            """
            Construct launch args before CompiledKernel.launch_metadata is added.
            """
            return (
                grid_0,
                grid_1,
                grid_2,
                num_warps,
                *cta_args,
                shared,
                stream,
                function,
                launch_enter_hook,
                launch_exit_hook,
                metadata,
            )

        # Getting the kernel launch args is extremely perf-sensitive.  Evaluating
        # `bin.launch_metadata` is relatively expensive, and returns None unless a
        # `launch_enter_hook` is installed.  So if we don't have that hook installed,
        # we want to burn None in to the launch args with zero overhead.
        # See https://github.com/pytorch/pytorch/issues/123597
        if binary.launch_enter_hook:

            def get_launch_args_with_kernel_launch_metadata(
                grid,
                grid_0,
                grid_1,
                grid_2,
                stream,
                function,
                metadata,
                bin,
                launch_enter_hook,
                launch_exit_hook,
                num_warps,
                shared,
                cta_args,
                args,
            ):
                """
                Construct launch args after CompiledKernel.launch_metadata is added
                by https://github.com/openai/triton/pull/3492 .
                """
                return (
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    bin.launch_metadata(grid, stream, *args),
                    launch_enter_hook,
                    launch_exit_hook,
                )

        else:

            def get_launch_args_with_kernel_launch_metadata(
                grid,
                grid_0,
                grid_1,
                grid_2,
                stream,
                function,
                metadata,
                bin,
                launch_enter_hook,
                launch_exit_hook,
                num_warps,
                shared,
                cta_args,
                args,
            ):
                """
                Construct launch args after CompiledKernel.launch_metadata is added
                by https://github.com/openai/triton/pull/3492 .
                """
                return (
                    grid_0,
                    grid_1,
                    grid_2,
                    stream,
                    function,
                    metadata,
                    None,
                    launch_enter_hook,
                    launch_exit_hook,
                )

        scope["get_launch_args"] = (
            get_launch_args_with_kernel_launch_metadata
            if hasattr(binary, "launch_metadata")
            else get_launch_args_without_kernel_launch_metadata
        )

        scope["runner"] = get_first_attr(binary, "run", "c_wrapper")

        exec(
            f"""
            def launcher({', '.join(def_args)}, grid, stream):
                if callable(grid):
                    grid_0, grid_1, grid_2 = grid(grid_meta)
                else:
                    grid_0, grid_1, grid_2 = grid

                args = {', '.join(call_args)},
                launch_args = get_launch_args(
                    grid, grid_0, grid_1, grid_2, stream, function,
                    metadata, bin, launch_enter_hook, launch_exit_hook,
                    num_warps, shared, cta_args, args
                )
                runner(*launch_args, *args)
                return bin
            """.lstrip(),
            scope,
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = self.fn
            launcher.bin = binary

        return binary, launcher

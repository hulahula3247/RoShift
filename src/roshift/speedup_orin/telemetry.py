"""Read Jetson telemetry without desktop NVIDIA management commands."""
from pathlib import Path
import time


def is_jetson():
    return Path('/etc/nv_tegra_release').is_file()


def capture():
    result = dict(platform='jetson', timestamp=time.time(), thermal={}, errors={})

    def read(path):
        try:
            return path.read_text().strip()
        except (OSError, TypeError, ValueError) as error:
            result['errors'][str(path)] = type(error).__name__
            return None

    result['meminfo'] = read(Path('/proc/meminfo'))
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        result['thermal'][zone.name] = dict(type=read(zone/'type'),
                                           millidegrees_c=read(zone/'temp'))
    result['gpu_devfreq'] = {}
    for gpu in Path('/sys/class/devfreq').glob('*gpu*'):
        result['gpu_devfreq'][gpu.name] = {
            key: read(gpu/key) for key in ('cur_freq', 'min_freq', 'max_freq', 'governor')}
    return result

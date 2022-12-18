import os
import logging
from dynaconf import Dynaconf

# specifying logging level
logging.basicConfig(level=logging.INFO)

current_directory = os.path.dirname(os.path.realpath(__file__))

settings = Dynaconf(
    envvar_prefix='DYNACONF',
    settings_files=[f'{current_directory}/settings.toml']
)

# settings = Dynaconf(settings_file='setting.toml',
#                     environments=True,
#                     envvar_prefix="DYNACONF")
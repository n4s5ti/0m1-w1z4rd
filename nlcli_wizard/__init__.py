"""
wiz4rd - isolated Omi flow-router implementation

The Python import namespace remains ``nlcli_wizard`` to preserve the upstream
training and evaluation substrate while the distribution and runtime identities
are isolated under the customized product name.
"""

__version__ = "0.1.0+omi.1"
__author__ = "Pranav Kumaar"

from nlcli_wizard.agent import NLCLIAgent
from nlcli_wizard.model import ModelManager

__all__ = ["NLCLIAgent", "ModelManager"]

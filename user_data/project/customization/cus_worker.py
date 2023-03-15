from typing import Any, Dict, Optional

from freqtrade.constants import Config
from freqtrade.customization.cus_freqtradebot import CusFreqtradeBot
from freqtrade.worker import Worker


class CusWorker(Worker):
    """Custom worker class that use customized trader bot

    Args:
        Worker (_type_): _description_
    """
    def __init__(self, args: Dict[str, Any], config: Optional[Config] = None) -> None:
        super().__init__(args, config)
        self.freqtrade = CusFreqtradeBot(config)

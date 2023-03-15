import sys

from freqtrade.configuration import Configuration
from freqtrade.optimize.backtesting import Backtesting


sys.path.append("/home/andy/CryptoTradingPlatform/freqtrade")

# Initialize empty configuration object
config = Configuration.\
    from_files(["/home/andy/CryptoTradingPlatform/freqtrade/\
        user_data/config.json"])
# Optionally (recommended), use existing configuration file
config["strategy"] = "SampleStrategy"
config["logfir"]
backtester = Backtesting(config)
backtester.start()

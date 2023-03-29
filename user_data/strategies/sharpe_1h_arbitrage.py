from sharpe_macd_arbitrage import SharpeMACDArbitrage
from base_arbitrage import TIMEFRAME_IN_MIN
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Sharpe1HArbitrage(SharpeMACDArbitrage):
    timeframe = "1h"
    custom_pair_number = 10
    custom_leverage_ratio = 2.0
    custom_take_profit_rate = 0.02
    custom_stop_loss_rate = -0.2
    custom_historic_preloaded_days  = 60
    custom_holding_period = 72
    custom_invest_rounds = 3
    custom_looking_back_period = 30 * 24

    minimal_roi = {
        "0": custom_take_profit_rate * custom_leverage_ratio,
        f"{custom_holding_period * TIMEFRAME_IN_MIN[timeframe]}":-1
    }
    stoploss = custom_stop_loss_rate * custom_leverage_ratio

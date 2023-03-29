from base_arbitrage import BASE_PAIR_NAME, TIMEFRAME_IN_MIN
import pandas as pd
from pandas import DataFrame
import logging
from typing import Tuple
from sharpe_macd_arbitrage import SharpeMACDArbitrage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXIT_DIFF_MEDIUM_THRESHOLD = 0
EXIT_DIFF_FAST_THRESHOLD = 0

class SharpeLessExitArbitrage(SharpeMACDArbitrage):
    timeframe = "1h"
    position_adjustment_enable = False
    use_exit_signal = True

   # Custom class variable not changing much
    custom_market_cycle = "SLOW"
    custom_pair_number = 10
    custom_leverage_ratio = 2.0
    custom_take_profit_rate = 0.02
    custom_stop_loss_rate = -0.1
    custom_historic_preloaded_days  = 30
    custom_holding_period = 7 * 6
    custom_invest_rounds = 3
    custom_looking_back_period = 14 * 24

    minimal_roi = {
        "0": custom_take_profit_rate * custom_leverage_ratio,
        f"{custom_holding_period * TIMEFRAME_IN_MIN[timeframe]}":-1
    }
    stoploss = custom_stop_loss_rate * custom_leverage_ratio

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        logger.debug("populate_exit_trend: IN")

        def get_exit_signal(pair:str,row:pd.Series):
            signal: Tuple[int,int] = (0,0)
            val_market = float(
                row.get(f"{BASE_PAIR_NAME}_{self.custom_market_cycle}_MACD",0))  #type:ignore

            if val_market != 0:
                if float(val_market) < 0:
                    diff_fast = float(row.get(f"{pair}_DIFF_FAST_BEAR",0))      #type:ignore
                    diff_medium = float(row.get(f"{pair}_DIFF_MEDIUM_BEAR",0))  #type:ignore
                else:
                    diff_fast = float(row.get(f"{pair}_DIFF_FAST_BULL",0))      #type:ignore
                    diff_medium = float(row.get(f"{pair}_DIFF_MEDIUM_BULL",0))  #type:ignore

                if (diff_fast > EXIT_DIFF_FAST_THRESHOLD) \
                    & (diff_medium > EXIT_DIFF_MEDIUM_THRESHOLD):
                    signal = (0,1)
                elif (diff_fast < -EXIT_DIFF_FAST_THRESHOLD) \
                    & (diff_medium < -EXIT_DIFF_MEDIUM_THRESHOLD):
                    signal = (1,0)

            if pair not in self.custom_exit_signal_dict.keys():
                self.custom_exit_signal_dict[pair] = dict()

            self.custom_exit_signal_dict[pair][row["date"]] = signal


        pair = metadata["pair"]
        shared_dataframe,_ = self.get_shared_analyzed_dataframe(self.timeframe)
        shared_dataframe.apply(
            lambda row: get_exit_signal(pair = pair,row = row), #type:ignore
            axis=1)

        dataframe["exit_long"] = dataframe.apply(
            lambda row: self.custom_exit_signal_dict[pair][row["date"]][0] \
                if row["date"] in self.custom_exit_signal_dict[pair].keys() else 0,
            axis=1
        )

        dataframe["exit_short"] = dataframe.apply(
            lambda row: self.custom_exit_signal_dict[pair][row["date"]][1] \
                if row["date"] in self.custom_exit_signal_dict[pair].keys() else 0,
            axis=1
        )

        logger.debug("populate_exit_trend: OUT")

        return dataframe

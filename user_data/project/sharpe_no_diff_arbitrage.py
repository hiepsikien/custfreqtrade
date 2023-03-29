from base_arbitrage import MACD_FAST, MACD_SIGNAL, MACD_SLOW,\
    BASE_PAIR_NAME, PERIOD_NUMBER_YEARLY, TIMEFRAME_IN_MIN
import pandas as pd
from typing import List
import numpy as np
import talib
import logging

from sharpe_macd_arbitrage import SharpeMACDArbitrage
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SharpeNoDiffArbitrage(SharpeMACDArbitrage):
    timeframe = "1h"
    position_adjustment_enable = False
    use_exit_signal = True

    # Custom class variable may vary
    custom_allow_buy_more:bool=False
    custom_rebalance_budget:bool=True
    custom_pair_number = 10
    custom_leverage_ratio = 2.0
    custom_take_profit_rate = 0.02
    custom_stop_loss_rate = -0.2
    custom_historic_preloaded_days  = 60
    custom_holding_period = 72
    custom_invest_rounds = 3
    custom_looking_back_period = 14 * 24

    minimal_roi = {
        "0": custom_take_profit_rate * custom_leverage_ratio,
        f"{custom_holding_period * TIMEFRAME_IN_MIN[timeframe]}":-1
    }
    stoploss = custom_stop_loss_rate * custom_leverage_ratio

    def analyze_dataframe(self, df: pd.DataFrame, pairs: List[str]):
        logger.debug("analyze_dataframe:IN")
        for pair in pairs:
            close = df[f"{pair}_close"]

            _,_,fast = talib.MACD(               #type: ignore
                close,
                fastperiod = self.custom_short_multiple * MACD_FAST,
                slowperiod = self.custom_short_multiple * MACD_SLOW,
                signalperiod = self.custom_short_multiple * MACD_SIGNAL
            )
            fast = np.array(fast) / close
            fast = fast.astype(np.float32)

            _,_,medium = talib.MACD(               #type: ignore
                close,
                fastperiod = self.custom_medium_multiple * MACD_FAST,
                slowperiod = self.custom_medium_multiple * MACD_SLOW,
                signalperiod = self.custom_medium_multiple * MACD_SIGNAL
            )
            medium = np.array(medium) / close
            medium = medium.astype(np.float32)

            _,_,slow = talib.MACD(                 #type: ignore
                close,
                fastperiod = self.custom_long_multiple * MACD_FAST,
                slowperiod = self.custom_long_multiple * MACD_SLOW,
                signalperiod = self.custom_long_multiple * MACD_SIGNAL
            )
            slow = slow / close
            slow = slow.astype(np.float32)

            pct_change = close.pct_change()
            pct_change = pct_change.astype(np.float32)

            add_df = pd.DataFrame(index=df.index,data={
                f"{pair}_FAST_MACD": fast,
                f"{pair}_MEDIUM_MACD": medium,
                f"{pair}_SLOW_MACD": slow,
                f"{pair}_PCT_CHANGE": pct_change,
            })
            df = pd.concat([df,add_df],axis=1,join="outer")
            del pct_change, add_df, slow, medium, fast

        df = df.sort_index(ascending=True)

        for pair in pairs:
            logger.debug(pair)

            mean_bear = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]<0]\
                [f"{pair}_PCT_CHANGE"].rolling(self.custom_looking_back_period).mean()
            std_bear = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]<0]\
                [f"{pair}_PCT_CHANGE"].rolling(self.custom_looking_back_period).std()
            sharpe_bear = mean_bear/std_bear * np.sqrt(PERIOD_NUMBER_YEARLY[self.timeframe])
            sharpe_bear = sharpe_bear.astype(np.float32)
            sharpe_bear.name = f"{pair}_SHARPE_BEAR"

            mean_bull = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]>0]\
                [f"{pair}_PCT_CHANGE"].rolling(self.custom_looking_back_period).mean()
            std_bull = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]>0]\
                [f"{pair}_PCT_CHANGE"].rolling(self.custom_looking_back_period).std()
            sharpe_bull = mean_bull/std_bull * np.sqrt(PERIOD_NUMBER_YEARLY[self.timeframe])
            sharpe_bull = sharpe_bull.astype(np.float32)
            sharpe_bull.name = f"{pair}_SHARPE_BULL"

            df = pd.concat([df,sharpe_bear, sharpe_bull],axis=1,join="outer")
            del sharpe_bear, sharpe_bull, mean_bear, mean_bull, std_bear, std_bull

        df = df.sort_index(ascending=True)

        return df

    def arbitrage_both_sides(
        self,
        row:pd.Series,
        pairs:List[str],
        cutoff:int
        ):

        is_bull = False
        long_pairs = short_pairs = []

        val = row.get(
            f"{BASE_PAIR_NAME}_{self.custom_market_cycle}_MACD",None)

        if val is None:
            return [], [], dict(), dict()

        if val > 0:         #type: ignore
            is_bull = True
        elif val < 0:       #type: ignore
            is_bull = False
        else:
            return [], [], dict(), dict()

        long_dict = dict()
        short_dict = dict()
        inv_long_dict = dict()
        inv_short_dict = dict()

        for pair in pairs:
            macd_fast = float(row[f"{pair}_FAST_MACD"])
            macd_medium = float(row[f"{pair}_MEDIUM_MACD"])
            if (is_bull):
                sharpe = float(row[f"{pair}_SHARPE_BULL"])
            else:
                sharpe = float(row[f"{pair}_SHARPE_BEAR"])

            if (sharpe > 0) & (macd_medium > 0) & (macd_fast > 0) :
                long_dict[pair] = sharpe
                inv_long_dict[sharpe] = pair
            elif (sharpe < 0) & (macd_medium < 0) & (macd_fast < 0):
                short_dict[pair] = sharpe
                inv_short_dict[sharpe] = pair

        sorted_inv_long_dict = sorted(inv_long_dict,reverse=True)
        long_pairs = [inv_long_dict[sorted_inv_long_dict[i]]\
            for i in range(min(len(inv_long_dict),cutoff))]

        sorted_inv_short_dict = sorted(inv_short_dict,reverse=False)
        short_pairs = [inv_short_dict[sorted_inv_short_dict[i]] \
            for i in range(min(len(inv_short_dict),cutoff))]

        return long_pairs, short_pairs, long_dict, short_dict

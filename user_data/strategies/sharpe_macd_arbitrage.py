from base_arbitrage import BaseArbitrage, MACD_FAST, \
    MACD_SIGNAL, MACD_SLOW, BASE_PAIR_NAME, PERIOD_NUMBER_YEARLY, TIMEFRAME_IN_MIN
import pandas as pd
from pandas import DataFrame
from typing import List, Optional, Tuple, Dict
import numpy as np
import talib
import logging
from datetime import datetime, timedelta
from freqtrade.persistence.models import Trade
from freqtrade.enums import RunMode
from freqtrade.constants import Config
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SharpeMACDArbitrage(BaseArbitrage):
    timeframe = "4h"
    position_adjustment_enable = False
    use_exit_signal = True

   # Custom class variable not changing much
    custom_market_cycle = "SLOW"
    custom_altcoin_cycle = "MEDIUM"
    custom_market_cycle_list = ["FAST","MEDIUM","SLOW"]

    # Custom class variable may vary
    custom_allow_buy_more:bool=False
    custom_rebalance_budget:bool=True
    custom_pair_number = 10
    custom_leverage_ratio = 2.0
    custom_take_profit_rate = 0.03
    custom_stop_loss_rate = -0.2
    custom_historic_preloaded_days  = 120
    custom_holding_period = 18
    custom_invest_rounds = 3
    custom_looking_back_period = 90 * 6

    minimal_roi = {
        "0": custom_take_profit_rate * custom_leverage_ratio,
        f"{custom_holding_period * TIMEFRAME_IN_MIN[timeframe]}":-1
    }
    stoploss = custom_stop_loss_rate * custom_leverage_ratio

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.custom_exit_signal_dict: Dict[str,Dict[datetime,Tuple[int,int]]]= dict()

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

        #Calculating return diff comparing to basecoin
        logger.info("...calculating percentage change diff compare to base pair")
        for pair in pairs:
            diff = df[f"{pair}_PCT_CHANGE"] - df[f"{BASE_PAIR_NAME}_PCT_CHANGE"]
            diff = diff.astype(np.float32)
            diff.name = f"{pair}_DIFF_PCT_CHANGE"
            df = pd.concat([df,diff],axis=1,join="outer")
            del diff

        df = df.sort_index(ascending=True)

        #Calculate sharpe and sharpe diff
        logger.info("...calculating Sharpe diff and MACD diff of all pairs...")
        for pair in pairs:
            logger.debug(pair)
            for cycle in self.custom_market_cycle_list:
                diff_bear = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]<0]\
                    [f"{pair}_{cycle}_MACD"]- df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]<0]\
                        [f"{BASE_PAIR_NAME}_{cycle}_MACD"]
                diff_bear = diff_bear.astype(np.float32)
                diff_bear.name = f"{pair}_DIFF_{cycle}_BEAR"

                mean_bear = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]<0]\
                    [f"{pair}_DIFF_PCT_CHANGE"].rolling(self.custom_looking_back_period).mean()
                std_bear = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]<0]\
                    [f"{pair}_DIFF_PCT_CHANGE"].rolling(self.custom_looking_back_period).std()
                sharpe_bear = mean_bear/std_bear * np.sqrt(PERIOD_NUMBER_YEARLY[self.timeframe])
                sharpe_bear = sharpe_bear.astype(np.float32)
                sharpe_bear.name = f"{pair}_{cycle}_SHARPE_BEAR"

                diff_bull = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]>0]\
                    [f"{pair}_{cycle}_MACD"] - df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]>0]\
                        [f"{BASE_PAIR_NAME}_{cycle}_MACD"]
                diff_bull = diff_bull.astype(np.float32)
                diff_bull.name = f"{pair}_DIFF_{cycle}_BULL"

                mean_bull = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]>0]\
                    [f"{pair}_DIFF_PCT_CHANGE"].rolling(self.custom_looking_back_period).mean()
                std_bull = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]>0]\
                    [f"{pair}_DIFF_PCT_CHANGE"].rolling(self.custom_looking_back_period).std()

                sharpe_bull = mean_bull/std_bull * np.sqrt(PERIOD_NUMBER_YEARLY[self.timeframe])
                sharpe_bull = sharpe_bull.astype(np.float32)
                sharpe_bull.name = f"{pair}_{cycle}_SHARPE_BULL"

                df = pd.concat([df,diff_bear, diff_bull, sharpe_bear,sharpe_bull],
                               axis=1,join="outer")
                del diff_bear, diff_bull, mean_bear, std_bear, mean_bull,
                std_bull, sharpe_bear, sharpe_bull

        df = df.sort_index(ascending=True)
        logger.debug("analyze_dataframe:OUT")

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        logger.debug("populate_entry_trend:IN")
        logger.info(f"populate_entry_trend: {metadata['pair']}")

        def _calculate_trade_signal(row):
            long_pairs, short_pairs, \
                _,_ = self.arbitrage_both_sides(
                row = row,
                pairs = self.dp.current_whitelist(),
                cutoff=self.custom_pair_number * 2
            )

            self.custom_entry_signal_dict[row["date"]] = (
                long_pairs,
                short_pairs
            )

        def _get_signal_long(date:datetime,pair:str):
            long_pairs = []
            if date in self.custom_entry_signal_dict.keys():
                long_pairs,_ = self.custom_entry_signal_dict[date]
            else:
                logger.debug(f"{date} not found as key in custom_entry_signal_dict")
            signal = 1 if pair in long_pairs else 0
            return signal

        def _get_signal_short(date:datetime,pair:str):
            short_pairs = []
            if date in self.custom_entry_signal_dict.keys():
                _, short_pairs = self.custom_entry_signal_dict[date]
            else:
                logger.debug(f"{date} not found as key in custom_entry_signal_dict")
            signal = 1 if pair in short_pairs else 0
            return signal

        end_date = dataframe.iloc[-1]["date"]

        if (end_date not in self.custom_entry_signal_dict.keys()):
            shared_dataframe,_ = self.get_shared_analyzed_dataframe(self.timeframe)
            if (shared_dataframe is None) or shared_dataframe.iloc[-1]["date"] != end_date:
                raise ValueError("last candle not matching")

            shared_dataframe.apply(_calculate_trade_signal,axis=1)  #type: ignore

        dataframe["enter_long"] = dataframe.apply(
            lambda row: _get_signal_long(row["date"],metadata["pair"]),
            axis=1
        )
        dataframe["enter_short"] = dataframe.apply(
            lambda row: _get_signal_short(row["date"],metadata["pair"]),
            axis=1
        )
        logger.debug("populate_entry_trend:OUT")
        return dataframe.copy()

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

        if val > 0:
            is_bull = True
        elif val < 0:
            is_bull = False
        else:
            return [], [], dict(), dict()

        long_dict = dict()
        short_dict = dict()
        inv_long_dict = dict()
        inv_short_dict = dict()

        for pair in pairs:
            if (is_bull):

                diff_fast = float(row[f"{pair}_DIFF_FAST_BULL"])
                diff_medium = float(row[f"{pair}_DIFF_MEDIUM_BULL"])
                sharpe = float(row[f"{pair}_{self.custom_altcoin_cycle}_SHARPE_BULL"])
            else:
                diff_fast = float(row[f"{pair}_DIFF_FAST_BEAR"])
                diff_medium = float(row[f"{pair}_DIFF_MEDIUM_BEAR"])
                sharpe = float(row[f"{pair}_{self.custom_altcoin_cycle}_SHARPE_BEAR"])

            if (sharpe > 0) & (diff_medium > 0) & (diff_fast > 0) :
                long_dict[pair] = sharpe
                inv_long_dict[sharpe] = pair
            elif (sharpe < 0) & (diff_medium < 0) & (diff_fast < 0):
                short_dict[pair] = sharpe
                inv_short_dict[sharpe] = pair

        sorted_inv_long_dict = sorted(inv_long_dict,reverse=True)
        long_pairs = [inv_long_dict[sorted_inv_long_dict[i]] \
            for i in range(min(len(inv_long_dict),cutoff))]

        sorted_inv_short_dict = sorted(inv_short_dict,reverse=False)
        short_pairs = [inv_short_dict[sorted_inv_short_dict[i]] \
            for i in range(min(len(inv_short_dict),cutoff))]

        long_dict[BASE_PAIR_NAME] = 0
        short_dict[BASE_PAIR_NAME] = 0

        if long_pairs and not short_pairs:
            short_pairs += [BASE_PAIR_NAME]
        elif short_pairs and not long_pairs:
            long_pairs += [BASE_PAIR_NAME]

        return long_pairs, short_pairs, long_dict, short_dict

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        logger.debug("populate_exit_trend: IN")

        def get_exit_signal(pair:str,row:pd.Series):
            signal: Tuple[int,int] = (0,0)
            val_market = float(row.get(f"{BASE_PAIR_NAME}_{self.custom_market_cycle}_MACD",0))

            if val_market != 0:
                if float(val_market) < 0:
                    val = float(row.get(f"{pair}_DIFF_FAST_BEAR",0))
                else:
                    val = float(row.get(f"{pair}_DIFF_FAST_BULL",0))

                if val > 0:
                    signal = (0,1)
                elif val < 0:
                    signal = (1,0)

            if pair not in self.custom_exit_signal_dict.keys():
                self.custom_exit_signal_dict[pair] = dict()

            self.custom_exit_signal_dict[pair][row["date"]] = signal

        pair = metadata["pair"]
        shared_dataframe,_ = self.get_shared_analyzed_dataframe(self.timeframe)
        shared_dataframe.apply(
            lambda row: get_exit_signal(pair = pair,row = row),
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

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """
        Customize stake size for each new trade.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_stake: A stake amount proposed by the bot.
        :param min_stake: Minimal stake size allowed by exchange.
        :param max_stake: Balance available for trading.
        :param leverage: Leverage selected for this trade.
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A stake size, which is between min_stake and max_stake.
        """
        open_pairs = [trade.pair for trade in Trade.get_open_trades()]

        if self.dp.runmode == RunMode.BACKTEST:
            latest_time = current_time - timedelta(minutes=TIMEFRAME_IN_MIN[self.timeframe])
        else:
            latest_time = self.get_shared_analyzed_dataframe(self.timeframe)[0].iloc[-1]["date"]

        if latest_time in self.custom_long_short_amount_dict.keys():
            long_stake, short_stake = self.custom_long_short_amount_dict[latest_time]
        else:
            if latest_time not in self.custom_entry_signal_dict.keys():
                return 0

            long_pairs, short_pairs = self.custom_entry_signal_dict[latest_time]

            for pair in open_pairs:
                if pair in long_pairs:
                    long_pairs.remove(pair)
                if pair in short_pairs:
                    short_pairs.remove(pair)

            invest_budget = min(
                self.wallets.get_total_stake_amount()/self.custom_invest_rounds, #type: ignore
                self.wallets.get_available_stake_amount() #type: ignore
            )

            if self.custom_rebalance_budget:
                long_sum, short_sum, _, _ \
                    = self.get_open_trades_info()
                long_budget = max(0,0.5 * (short_sum - long_sum + invest_budget))
                short_budget = invest_budget - long_budget
            else:
                long_budget = short_budget = invest_budget/2

            long_number = min(len(long_pairs),self.custom_pair_number)
            short_number = min(len(short_pairs),self.custom_pair_number)

            if min(long_number,short_number) == 0:
                long_number = short_number = 0

            long_stake = long_budget/long_number if long_number > 0 else 0
            short_stake = short_budget/short_number if short_number > 0 else 0

            self.custom_long_short_amount_dict[latest_time] = (long_stake,short_stake)

        if (not self.custom_allow_buy_more) and (pair in open_pairs):
            return None

        if side == "long":
            return long_stake
        else:
            return short_stake

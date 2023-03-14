
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
from re import I
import sys
from freqtrade.constants import Config
sys.path.append("/home/andy/CryptoTradingPlatform/freqtrade")

from datetime import datetime
import numpy as np  # noqaclear
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, Union, List, Tuple, Dict
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib as ta
import logging
from freqtrade.persistence.trade_model import LocalTrade, Trade
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# This class is a sample. Feel free to customize it.

# Market related parameters
MACD_FAST = 26
MACD_SLOW= 12
MACD_SIGNAL= 9
LONG_MULTIPLES = 8
MEDIUM_MULTIPLES = 3
SHORT_MULTIPLES = 1
MARKET_CYCLES = ["MEDIUM","SLOW"]
BASE_PAIR_NAME = "BTC/USDT:USDT"
MARKET_CYCLE = "SLOW"
ALTCOIN_CYCLE = "MEDIUM"

#Trading related parameters
TIMEFRAME = "4h"
PAIR_NUMBER = 3
INVESTMENT_TRUNK = 7
HOLDING_PERIOD_IN_MIN = int(2 * 24 * 60)
LEVERAGE_RATIO = 2.0
TAKE_PROFIT = 0.6
STOP_LOSS = -0.3
STARTUP_CANDLE_COUNT = 30 * 6

class ArbitrageBothSides(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": TAKE_PROFIT,
        f"{HOLDING_PERIOD_IN_MIN}":-1
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = STOP_LOSS

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = TIMEFRAME

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = STARTUP_CANDLE_COUNT

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # Allow trade position adjustment:
    position_adjustment_enable = True

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._cached_indicator_time: Optional[str] = None
        self._cached_indicator_data: Optional[pd.DataFrame] = None
        self._cached_signal_time: Optional[str]= None
        self._cached_signal_data: Optional[pd.DataFrame] = None
        self._cached_stake_time:Optional[str] = None
        self._cached_stake_data:Optional[pd.DataFrame] = None
        self._cached_adjust_position = dict()

    def informative_pairs(self):
        logger.info("information_pairs:IN")
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, self.timeframe) for pair in pairs]
        logger.info("information_pairs:OUT")
        return informative_pairs

    def _calculate_macd(self, df:pd.DataFrame,pairs:List[str]):
        print("...calculating MACD ...")
        for pair in pairs:
            _,_,medium = ta.MACD( #type: ignore
                df[f"{pair}_close"],
                fastperiod = MEDIUM_MULTIPLES * MACD_FAST,
                slowperiod = MEDIUM_MULTIPLES * MACD_SLOW,
                signalperiod = MEDIUM_MULTIPLES * MACD_SIGNAL
            )

            medium = np.array(medium) / df[f"{pair}_close"]

            _,_,slow = ta.MACD( #type: ignore
                df[f"{pair}_close"],
                fastperiod = LONG_MULTIPLES * MACD_FAST,
                slowperiod = LONG_MULTIPLES * MACD_SLOW,
                signalperiod = LONG_MULTIPLES * MACD_SIGNAL
            )

            slow = slow / df[f"{pair}_close"]

            add_df = pd.DataFrame(index=df.index,data={
                f"{pair}_MEDIUM_MACD" : medium,
                f"{pair}_SLOW_MACD" : slow
            })

            df = pd.concat([df,add_df],axis=1)

        df.sort_index(ascending=True,inplace=True)
        df.fillna(method="ffill", inplace = True)

        return df

    def _calculate_macd_diff_with_base_pair(self, df:pd.DataFrame,pairs:List[str]):
        print("...calculating MACD diff between altcoins and BTC...")

        #Calculating diff
        for pair in pairs:
            for cycle in MARKET_CYCLES:

                diff = df[f"{pair}_{cycle}_MACD"] - df[f"{BASE_PAIR_NAME}_{cycle}_MACD"]
                diff.name = f"{pair}_DIFF_{cycle}"

                diff_bear = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]<0][f"{pair}_{cycle}_MACD"] \
                    - df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]<0][f"{BASE_PAIR_NAME}_{cycle}_MACD"]
                diff_bear.name = f"{pair}_DIFF_{cycle}_BEAR"

                diff_bull = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]>0][f"{pair}_{cycle}_MACD"] \
                - df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]>0][f"{BASE_PAIR_NAME}_{cycle}_MACD"]
                diff_bull.name = f"{pair}_DIFF_{cycle}_BULL"

                df = pd.concat([df,diff,diff_bear,diff_bull],axis=1)

        return df

    def _calculate_macd_diff_mean(self,df:pd.DataFrame,pairs:list[str]):
        #Calculating rolling diff mean
        print("...calculating MACD diff mean...")
        for pair in pairs:
            for cycle in MARKET_CYCLES:
                diff_bear_mean = df[f"{pair}_DIFF_{cycle}_BEAR"].expanding().mean()
                diff_bear_mean.name = f"{pair}_DIFF_{cycle}_BEAR_MEAN"
                diff_bull_mean = df[f"{pair}_DIFF_{cycle}_BULL"].expanding().mean()
                diff_bull_mean.name = f"{pair}_DIFF_{cycle}_BULL_MEAN"
                df = pd.concat([df,diff_bear_mean,diff_bull_mean],axis=1)
        return df

    def foo(self, row):
        longs, shorts = self.arbitrate_both_sides(
            row = row,
            pairs = self.dp.current_whitelist()
        )
        return pd.Series(
            {
                "longs": longs,
                "shorts": shorts
            }
        )

    def calculate_signal_for_all_pairs(self, analyzed_df:pd.DataFrame):

        df = analyzed_df.apply(self.foo,axis=1)
        df["date"] = analyzed_df["date"]
        for pair in self.dp.current_whitelist():
            df[f"{pair}_LONG"] = df.apply(
                lambda row: 1 if pair in row["longs"] else 0,
                axis=1
            )
            df[f"{pair}_SHORT"] = df.apply(
                lambda row: 1 if pair in row["shorts"] else 0,
                axis=1
            )

        return df.copy()

    def arbitrate_both_sides(self,
                            row:pd.Series,
                            pairs:List[str]
                            ):

        is_bull: bool = False
        long_pairs: List[str] = []
        short_pairs: List[str] = []

        if row[f"{BASE_PAIR_NAME}_{MARKET_CYCLE}_MACD"] > 0:
            is_bull = True

        long_dict = dict()
        short_dict = dict()
        inv_long_dict = dict()
        inv_short_dict = dict()

        for pair in pairs:
            if (is_bull):
                diff = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE}_BULL"])
                diff_mean = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE}_BULL_MEAN"])
            else:
                diff = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE}_BEAR"])
                diff_mean = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE}_BEAR_MEAN"])

            if (diff_mean > 0) & (diff > 0):
                long_dict[pair] = diff_mean
                inv_long_dict[diff_mean] = pair
            elif (diff_mean < 0) & (diff < 0):
                short_dict[pair] = diff_mean
                inv_short_dict[diff_mean] = pair

        sorted_inv_long_dict = sorted(inv_long_dict,reverse=True)
        long_pairs = [inv_long_dict[sorted_inv_long_dict[i]] for i in range(min(len(long_dict),PAIR_NUMBER))]

        sorted_inv_short_dict = sorted(inv_short_dict,reverse=False)
        short_pairs = [inv_short_dict[sorted_inv_short_dict[i]] for i in range(min(len(short_dict),PAIR_NUMBER))]

        if long_pairs and not short_pairs:
            short_pairs += [BASE_PAIR_NAME]
            short_dict[BASE_PAIR_NAME] = 0
        elif short_pairs and not long_pairs:
            long_pairs += [BASE_PAIR_NAME]
            long_dict[BASE_PAIR_NAME] = 0

        return long_pairs, short_pairs

    def get_trade_signal(
        self,
        row:pd.Series,
        current_pair:str,
        pairs:list[str]
        ):

        long_pairs, short_pairs = self.arbitrate_both_sides(
                row = row,
                pairs=pairs
        )

        return self.get_signal_from_lists(
            current_pair=current_pair,
            long_pairs=long_pairs,
            short_pairs=short_pairs
        )

    def get_signal_from_lists(self,current_pair,long_pairs,short_pairs):
        long_signal = 0
        short_signal = 0

        if current_pair in long_pairs:
            long_signal = 1
            # logger.info(f"long_pairs = {long_pairs}")
            # logger.info(f"long_dict = {long_dict}")
            # logger.info(f"{row['date']}: LONG {current_pair}: diff_mean = {long_dict[current_pair]}")

        if current_pair in short_pairs:
            short_signal = 1
            # logger.info(f"short_pairs = {short_pairs}")
            # logger.info(f"short_dict = {short_dict}")
            # logger.info(f"{row['date']}: SHORT {current_pair}: diff_mean = {short_dict[current_pair]}")

        return long_signal, short_signal

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # logger.info("populate_indicators: IN")

        latest_date = dataframe.iloc[-1]["date"]

        if (self._cached_indicator_time == latest_date) & (self._cached_indicator_data is not None):
            # logger.info("return cached dataframe")
            dataframe = dataframe.merge(self._cached_indicator_data,on="date")  #type: ignore
            return dataframe

        pairs = self.dp.current_whitelist()
        # logger.info(f"whitelist_pairs: {pairs}")

        for pair in self.dp.current_whitelist():
            informative:pd.DataFrame= self.dp.get_pair_dataframe(
                pair = pair,
                timeframe = self.timeframe
            )
            informative = informative[["date","close"]]
            informative = informative.rename(columns = {"close":f"{pair}_close"})

            dataframe = dataframe.merge(informative,on="date")

        dataframe = self._calculate_macd(df=dataframe,pairs=pairs)
        dataframe = self._calculate_macd_diff_with_base_pair(df=dataframe,pairs=pairs)
        dataframe = self._calculate_macd_diff_mean(df=dataframe,pairs=pairs)

        self._cached_indicator_time = dataframe.iloc[-1]["date"]
        self._cached_indicator_data = dataframe.drop(["open","high","low","close","volume"],axis=1)

        # logger.info("populate_indicators: OUT")

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        # logger.info("populate_entry_trend: IN")
        logger.info(f"populate_entry_trend: {metadata['pair']}")

        latest = dataframe.iloc[-1]["date"]

        df: Optional[pd.DataFrame] = self._cached_signal_data
        if (latest != self._cached_signal_time) | (self._cached_signal_data is None):
            df = self.calculate_signal_for_all_pairs(dataframe)
            self._cached_signal_data = df
            self._cached_signal_time = latest

        pair = metadata["pair"]
        dataframe["enter_long"] = df[f"{pair}_LONG"]    #type: ignore
        dataframe["enter_short"] = df[f"{pair}_SHORT"]  #type: ignore

        # dataframe["enter_long"], dataframe["enter_short"] =  dataframe.apply(lambda row: self.get_trade_signal(
        #     row=row,
        #     current_pair=metadata["pair"],
        #     pairs = self.dp.current_whitelist()
        #     ),
        #     result_type="expand",
        #     axis=1
        # )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        return dataframe

    def bot_start_loop(self):
        logger.info("bot_start_loop: IN")
        logger.info("bot_start_loop: OUT")

    def bot_start(self):
        logger.info("bot_start: IN")
        logger.info("bot_start: OUT")


    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        """ We will enter the trades with the amount that will help to balanceing long and short,
        given current open trades number and side, also the number of trade may open in this period
        depend on the signal.

        Returns:
            float: stake amount, invest amount
        """
        # logger.info("custom_stake_amount: IN")

        n_current_long = 0
        n_current_short = 0
        sum_current_long = 0
        sum_current_short = 0

        open_pairs: List[str]= []

        for trade in LocalTrade.trades_open:
            if trade.amount>0:
                if trade.is_short:
                    n_current_short += 1
                    sum_current_short += trade.amount
                else:
                    n_current_long += 1
                    sum_current_long += trade.amount
                open_pairs.append(trade.pair)

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        current_candle = dataframe.iloc[-1]

        long_pairs, short_pairs = self.arbitrate_both_sides(
            row = current_candle,
            pairs=self.dp.current_whitelist()
        )

        n_new_long = len(long_pairs)
        # for pair in long_pairs:
        #     if pair in open_pairs:
        #         n_new_long += -1

        n_new_short = len(short_pairs)
        # for pair in short_pairs:
        #     if pair in open_pairs:
        #         n_new_short += -1

        available_stake_amount = self.wallets.get_available_stake_amount() #type: ignore
        invest_trunk = self.wallets.get_total_stake_amount()/INVESTMENT_TRUNK #type: ignore
        invest_budget = min(available_stake_amount,invest_trunk)

        long_stake_total = invest_budget/2
        short_stake_total = invest_budget/2

        long_stake = long_stake_total/n_new_long if n_new_long > 0 else 0
        short_stake = short_stake_total/n_new_short if n_new_short > 0 else 0

        result_stake = long_stake if side == "long" else short_stake

        #Update adjust position dictionary
        for pair in long_pairs:
            if pair in open_pairs:
                if pair not in self._cached_adjust_position.keys():
                    self._cached_adjust_position[pair] = dict()
                self._cached_adjust_position[pair][current_candle["date"]] = (pair,"long",result_stake)

        for pair in short_pairs:
            if pair in open_pairs:
                if pair not in self._cached_adjust_position.keys():
                    self._cached_adjust_position[pair] = dict()
                self._cached_adjust_position[pair][current_candle["date"]] = (pair,"short",result_stake)

        return result_stake

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Customize leverage for each new trade. This method is only called in futures mode.

        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        return LEVERAGE_RATIO

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:
        """
        Custom trade adjustment logic, returning the stake amount that a trade should be
        increased or decreased.
        This means extra buy or sell orders with additional fees.
        Only called when `position_adjustment_enable` is set to True.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns None

        :param trade: trade object.
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Current buy rate.
        :param current_profit: Current profit (as ratio), calculated based on current_rate.
        :param min_stake: Minimal stake size allowed by exchange (for both entries and exits)
        :param max_stake: Maximum stake allowed (either through balance, or by exchange limits).
        :param current_entry_rate: Current rate using entry pricing.
        :param current_exit_rate: Current rate using exit pricing.
        :param current_entry_profit: Current profit using entry pricing.
        :param current_exit_profit: Current profit using exit pricing.
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return float: Stake amount to adjust your trade,
                       Positive values to increase position, Negative values to decrease position.
                       Return None for no action.
        """
        logger.debug("adjust_trade_position:IN")

        pair: str = trade.pair

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_date = dataframe.iloc[-1]["date"]

        adjust_position_dict: Optional[Tuple] = self._cached_adjust_position[pair] \
            if pair in self._cached_adjust_position.keys() else None

        if (adjust_position_dict):
            if current_date in adjust_position_dict.keys():
                (adj_pair,side,amount) = adjust_position_dict[current_date]

                if (pair == adj_pair):
                    if (trade.is_short == (side=="short")):
                        # logger.info(f"Asked to increase {pair} by amount {amount}")
                        result_amount = amount
                    else:
                        if amount < trade.stake_amount:
                            # logger.info(f"Asked to decrease {pair} by amount {amount}")
                            result_amount = -amount
                        else:
                            # logger.info(f"Want to open oppsotive position for {pair} by amount {amount - trade.stake_amount}")
                            result_amount = trade.stake_amount
                    return result_amount

        logger.debug("adjust_trade_position:OUT")
        return None

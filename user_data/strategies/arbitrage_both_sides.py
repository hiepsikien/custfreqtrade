
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import sys

from matplotlib.style import available
from freqtrade.constants import Config
from freqtrade.exchange.exchange_utils import amount_to_contract_precision
sys.path.append("/home/andy/CryptoTradingPlatform/freqtrade")

from datetime import datetime, timedelta
import numpy as np  # noqaclear
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, List, Tuple, Dict
from freqtrade.strategy import IStrategy

# --------------------------------
# Add your lib to import here
# import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib as ta
import logging
from freqtrade.persistence.trade_model import Trade
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import psutil
import os

# This class is a sample. Feel free to customize it.

# Market related parameters
VERSION = "1.0"
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
TRADING_TIMEFRAME_IN_MIN = 4 * 60
PAIR_NUMBER = 5
HOLDING_PERIOD_IN_TIMEFRAMES = 3 * 6
INVEST_ROUNDS = HOLDING_PERIOD_IN_TIMEFRAMES / 2
LEVERAGE_RATIO = 2.0
TAKE_PROFIT_RATE = 0.5
STOP_LOSS_RATE = -0.25
STARTUP_CANDLE_COUNT = 365 * 6
TIMEFRAME_IN_MIN = {
    "1m":1,
    "5m":5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 4 * 60,
    "12h": 12 * 60,
    "1d": 24 * 60
}
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

    #Hyperotable parameter
    # holding_period_in_timeframe = CategoricalParameter([9,10,11,12,13,14,15,16,17,18],default=12,space="roi")

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": TAKE_PROFIT_RATE * LEVERAGE_RATIO,
        f"{HOLDING_PERIOD_IN_TIMEFRAMES * TRADING_TIMEFRAME_IN_MIN}":-1
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = STOP_LOSS_RATE * LEVERAGE_RATIO

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
        """ Overided initialization

        Attributes:
        - _cached_indicator_time: latest time that indicator data cached
        - _cached_indicator_data: cached indicator data
        - _cached_singal_time: latest time that entry signal data caached
        - _cached_signal_data: cached entry signal data
        - _cached_stake_time: latest time that stake amount computed
        - _cached_long_stake: cached stake amount for long positions
        - _cached_short_stake: cached stake amount for short positions
        - _cached_adjust_position: cached dictionary data related to adjust position

        Args:
            config (Config): _description_
        """
        super().__init__(config)
        self._cached_indicator_time: Optional[str] = None
        self._cached_indicator_data: Optional[pd.DataFrame] = None
        self._cached_signal_time: Optional[str]= None
        self._cached_signal_data: Optional[pd.DataFrame] = None
        self._cached_stake_time:Optional[datetime] = None
        self._cached_long_stake:Optional[float] = None
        self._cached_short_stake:Optional[float] = None
        self._cached_adjust_position = dict()

        self._long_pairs:Dict[datetime,List[str]] = dict()
        self._short_pairs:Dict[datetime,List[str]] = dict()

    def informative_pairs(self):
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
        logger.debug("information_pairs:IN")
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, self.timeframe) for pair in pairs]
        logger.debug("information_pairs:OUT")
        return informative_pairs

    def _calculate_macd(self, df:pd.DataFrame,pairs:List[str]):
        """ Calculate MACD indicator

        Args:
            df (pd.DataFrame): input dataframe
            pairs (List[str]): list of informative pairs

        Returns:
            pd:DataFrame: process dataframe
        """
        logger.debug("...calculating MACD ...")
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
        """ Added difference between pairs with base pair
        Args:
            df (pd.DataFrame): input dataframe
            pairs (List[str]): list of pairs

        Returns:
            pd.DataFrame: processed dataframe
        """
        logger.debug("...calculating MACD diff between altcoins and BTC...")
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
        """ Add mean of different between pair and the base pair
        Args:
            df (pd.DataFrame): input data
            pairs (list[str]): list of pairs

        Returns:
            pd.DataFrame: processed dataframe
        """
        logger.info("...calculating MACD diff mean...")
        for pair in pairs:
            for cycle in MARKET_CYCLES:
                diff_bear_mean = df[f"{pair}_DIFF_{cycle}_BEAR"].expanding().mean()
                diff_bear_mean.name = f"{pair}_DIFF_{cycle}_BEAR_MEAN"
                diff_bull_mean = df[f"{pair}_DIFF_{cycle}_BULL"].expanding().mean()
                diff_bull_mean.name = f"{pair}_DIFF_{cycle}_BULL_MEAN"
                df = pd.concat([df,diff_bear_mean,diff_bull_mean],axis=1)
        return df

    def foo(self, row):
        long_pairs, short_pairs = self.arbitrate_both_sides(
            row = row,
            pairs = self.dp.current_whitelist()
        )

        self._long_pairs[row["date"]] = long_pairs
        self._short_pairs[row["date"]] = short_pairs

        return pd.Series(
            {
                "longs": long_pairs,
                "shorts": short_pairs
            }
        )

    def calculate_signal_for_all_pairs(self, analyzed_df:pd.DataFrame):
        """ Calculate entry signal for all pairs

        Args:
            analyzed_df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: processed dataframe
        """
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
        df = df.copy()

        return df

    def arbitrate_both_sides(self,
                            row:pd.Series,
                            pairs:List[str]
                            ):
        """ Core algorithm function that compute set of pairs to go long
        or short at each timeframe.

        Args:
            row (pd.Series): current info
            pairs (List[str]): list of pairs to trade

        Returns:
            Tuple[[List[str],List[str]): a tuple of long pairs and short pairs
        """
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

    def get_signal_from_lists(self,current_pair:str,long_pairs:List[str],short_pairs:List[str]):
        """ Extract signal for a specific pair

        Args:
            current_pair (str): the pair
            long_pairs (List[str]): list of pairs to go long
            short_pairs (List[str]): list of pairs to go short

        Returns:
            Tuple[int,int]: a tuple such as (1,0) that say go long
        """
        long_signal = 0
        short_signal = 0

        if current_pair in long_pairs:
            long_signal = 1

        if current_pair in short_pairs:
            short_signal = 1

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
        latest_date = dataframe.iloc[-1]["date"]

        if (self._cached_indicator_time == latest_date) & (self._cached_indicator_data is not None):
            dataframe = dataframe.merge(self._cached_indicator_data,on="date")  #type: ignore
            return dataframe

        pairs = self.dp.current_whitelist()

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

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        logger.info(f"populate_entry_trend: {metadata['pair']}")

        latest = dataframe.iloc[-1]["date"]

        dataframe = dataframe.copy()

        df: Optional[pd.DataFrame] = self._cached_signal_data
        if (latest != self._cached_signal_time) | (self._cached_signal_data is None):
            df = self.calculate_signal_for_all_pairs(dataframe)
            self._cached_signal_data = df
            self._cached_signal_time = latest

        pair = metadata["pair"]
        dataframe["enter_long"] = df[f"{pair}_LONG"]    #type: ignore
        dataframe["enter_short"] = df[f"{pair}_SHORT"]  #type: ignore

        return dataframe.copy()

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        return dataframe

    def version(self):
        """ Get version

        Returns:
            _type_: _description_
        """
        return VERSION

    def bot_loop_start(self):
        """ Thing to do at bot loop start
        """
        logger.debug("bot_loop_start: IN")
        self.process_at_loop_start(None)    #type: ignore
        logger.debug("bot_loop_start: OUT")

    def bot_start(self):
        """Thing to do at bot start
        """
        logger.debug("bot_start: IN")
        logger.debug("bot_start: OUT")

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
        # logger.debug("custom_stake_amount: IN")

        open_pairs = [trade.pair for trade in Trade.get_open_trades()]
        latest = current_time-timedelta(minutes=TRADING_TIMEFRAME_IN_MIN)
        if self._cached_stake_time == latest:
            long_stake = self._cached_long_stake
            short_stake = self._cached_short_stake
        else:
            long_pairs = self._long_pairs[latest]
            short_pairs = self._short_pairs[latest]

            # logger.info(">TO LONG:{}".format(long_pairs))
            # logger.info(">TO SHORT:{}".format(short_pairs))

            invest_budget = min(
                self.wallets.get_total_stake_amount()/INVEST_ROUNDS, #type: ignore
                self.wallets.get_available_stake_amount() #type: ignore
            )
            long_sum, short_sum, _, _ \
                = self.get_open_trades_info()

            long_budget = max(0,0.5 * (short_sum - long_sum + invest_budget))
            short_budget = invest_budget - long_budget

            long_stake = long_budget/len(long_pairs) if len(long_pairs) > 0 else 0
            short_stake = short_budget/len(short_pairs) if len(short_pairs) > 0 else 0

            self._cached_long_stake = long_stake
            self._cached_short_stake = short_stake
            self._cached_stake_time = latest

            #Update adjust position dictionary
            for pair in long_pairs:
                if pair in open_pairs:
                    if pair not in self._cached_adjust_position.keys():
                        self._cached_adjust_position[pair] = dict()
                    if latest not in self._cached_adjust_position[pair].keys():
                        self._cached_adjust_position[pair][latest] = ("long",long_stake)

            for pair in short_pairs:
                if pair in open_pairs:
                    if pair not in self._cached_adjust_position.keys():
                        self._cached_adjust_position[pair] = dict()
                    if latest not in self._cached_adjust_position[pair].keys():
                            self._cached_adjust_position[pair][latest] = ("short",short_stake)

        return long_stake if side == "long" else short_stake    #type: ignore

    def get_open_trades_info(self):
        """
        Get info of open trades

        Returns:
            _type_: _description_
        """
        long_sum = 0
        short_sum = 0
        long_num = 0
        short_num = 0
        for trade in Trade.get_open_trades():
            if trade.trade_direction == "long":
                long_sum+=trade.stake_amount
                long_num+=1
            else:
                short_sum+=trade.stake_amount
                short_num+=1
        return long_sum, short_sum,long_num, short_num

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

    def backtest_loop_start_callback(self,current_time:datetime):
        """ The callback function that backtester call at the beginning of each timeframe
        Args:
            current_time (datetime): current time
        """
        self.process_at_loop_start(current_time)


    def process_at_loop_start(self,current_time:datetime):
        """
        Thing to do at the beginning of each loop.

        Args:
            current_time (datetime): current time
        """
        logger.debug("process_at_loop_start:IN")
        long_sum, short_sum,_,_ = self.get_open_trades_info()
        current_stake = long_sum + short_sum
        long_stake_ratio = round(long_sum/(long_sum+short_sum) if (long_sum+short_sum)>0 else 0,2)
        available_stake = self.wallets.get_available_stake_amount()    #type: ignore
        stake_usage = round(current_stake/(current_stake + available_stake),2)
        total_stake_value = int(self.wallets.get_total_stake_amount()) #type: ignore

        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss/1_000_000)

        logger.info(f"{current_time}| Long pct: {long_stake_ratio}| Used pct: "\
            f"{stake_usage}| Total: {total_stake_value}USDT| Mem: {mem_usage}M")

        logger.info(f"Available stake amount:{self.wallets.get_available_stake_amount()}")  #type:ignore
        logger.info(f"Total stake amount:{self.wallets.get_total_stake_amount()}")  #type:ignore

        #Print current position:
        logger.info("Current open positions:")
        long_open_trades = []
        short_open_trades = []
        for trade in Trade.get_open_trades():
            if trade.trade_direction == "long":
                long_open_trades.append((trade.pair,f"{trade.stake_amount}{trade.stake_currency}"))
            else:
                short_open_trades.append((trade.pair,f"{trade.stake_amount}{trade.stake_currency}"))

        logger.info("OPEN LONG: {}".format(long_open_trades))
        logger.info("OPEN SHORT: {}".format(short_open_trades))
        #Print entry signal
        latest_time = current_time - timedelta(minutes=TIMEFRAME_IN_MIN[TIMEFRAME])
        long_pairs = self._long_pairs[latest_time]
        short_pairs = self._short_pairs[latest_time]
        logger.info("TO LONG:{}".format(long_pairs))
        logger.info("TO SHORT:{}".format(short_pairs))

        logger.debug("process_at_loop_start:OUT")

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            side: str, **kwargs) -> bool:
        """
        Called right before placing a entry order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair that's about to be bought/shorted.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in target (base) currency that's going to be traded.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param current_time: datetime object, containing the current datetime
        :param entry_tag: Optional entry_tag (buy_tag) if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True is returned, then the buy-order is placed on the exchange.
            False aborts the process
        """
        logger.info(f"Placed {side} entry of {amount}{pair}")
        return True

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """
        Called right before placing a regular exit order.
        Timing for this function is critical, so avoid doing heavy computations or
        network requests in this method.

        For full documentation please go to https://www.freqtrade.io/en/latest/strategy-advanced/

        When not implemented by a strategy, returns True (always confirming).

        :param pair: Pair for trade that's about to be exited.
        :param trade: trade object.
        :param order_type: Order type (as configured in order_types). usually limit or market.
        :param amount: Amount in base currency.
        :param rate: Rate that's going to be used when using limit orders
                     or current rate for market orders.
        :param time_in_force: Time in force. Defaults to GTC (Good-til-cancelled).
        :param exit_reason: Exit reason.
            Can be any of ['roi', 'stop_loss', 'stoploss_on_exchange', 'trailing_stop_loss',
                           'exit_signal', 'force_exit', 'emergency_exit']
        :param current_time: datetime object, containing the current datetime
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        :return bool: When True, then the exit-order is placed on the exchange.
            False aborts the process
        """
        logger.info(f"Placed exit of {amount}{pair}")

        return True

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
        pair: str = trade.pair

        latest = current_time - timedelta(minutes=TRADING_TIMEFRAME_IN_MIN)

        adjust_position_dict: Optional[Tuple] = self._cached_adjust_position[pair] \
            if pair in self._cached_adjust_position.keys() else None

        if (adjust_position_dict is None):
            return None

        if latest not in adjust_position_dict.keys(): #type:ignore
            return None

        if adjust_position_dict[latest] == -1:  #type:ignore
            return None

        (side,amount) = adjust_position_dict[latest]  #type:ignore
        adjust_position_dict[latest] = -1 #type:ignore

        if (trade.trade_direction == side):
            logger.info(f"Asked to increase {side} {pair} for {amount} {trade.stake_currency}")
            result_amount = amount
        else:
            logger.info(f"Asked to decrease {side} {pair} for {amount} {trade.stake_currency}")
            if amount < trade.stake_amount:
                result_amount = -amount
            else:
                result_amount = trade.stake_amount
        return result_amount

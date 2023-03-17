
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import sys
from turtle import clear
from freqtrade import data
from freqtrade.constants import Config
from freqtrade.data.converter import trim_dataframe
from freqtrade.data.dataprovider import MAX_DATAFRAME_CANDLES
sys.path.append("/home/andy/CryptoTradingPlatform/freqtrade")

from datetime import datetime
import numpy as np  # noqaclear
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, List, Tuple, Dict
from freqtrade.strategy import IStrategy
from freqtrade.enums import CandleType, RunMode
from datetime import timezone, timedelta
from freqtrade.data.dataprovider import MAX_DATAFRAME_CANDLES
from freqtrade.data.converter import trim_dataframe

# --------------------------------
# Add your lib to import here
# import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib as ta
import logging
from freqtrade.persistence.trade_model import Trade
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from freqtrade.configuration import TimeRange, timerange
# This class is a sample. Feel free to customize it.

# Market related parameters
VERSION = "1.1.0"
MACD_FAST = 26
MACD_SLOW= 12
MACD_SIGNAL= 9
LONG_MULTIPLES = 8
MEDIUM_MULTIPLES = 3
SHORT_MULTIPLES = 1
MARKET_CYCLES = ["FAST","MEDIUM","SLOW"]
BASE_PAIR_NAME = "BTC/USDT:USDT"
MARKET_CYCLE = "SLOW"
ALTCOIN_CYCLE_FAST = "FAST"
ALTCOIN_CYCLE = "MEDIUM"

#Trading related parameters
TIMEFRAME = "30m"
TRADING_TIMEFRAME_IN_MIN = 4 * 60
PAIR_NUMBER = 3
HOLDING_PERIOD_IN_TIMEFRAMES = 2 * 24
INVEST_ROUNDS = HOLDING_PERIOD_IN_TIMEFRAMES / 2
LEVERAGE_RATIO = 2.0
TAKE_PROFIT_RATE = 0.07
STOP_LOSS_RATE = -0.05
STARTUP_CANDLE_COUNT = MACD_FAST * LONG_MULTIPLES

class NewArbitrage(IStrategy):
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
        self._cached_signal_time: Optional[str]= None
        self._cached_signal_data: Optional[pd.DataFrame] = None
        self._cached_stake_time:Optional[str] = None
        self._cached_long_stake:Optional[float] = None
        self._cached_short_stake:Optional[float] = None
        self._cached_adjust_position = dict()
        self._cached_shared_data: Dict[Tuple[str,CandleType], Tuple[DataFrame, datetime]] = {}

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
            _,_,fast = ta.MACD(                 #type: ignore
                df[f"{pair}_close"],
                fastperiod = SHORT_MULTIPLES * MACD_FAST,
                slowperiod = SHORT_MULTIPLES * MACD_SLOW,
                signalperiod = SHORT_MULTIPLES * MACD_SIGNAL
            )
            fast = fast/ df[f"{pair}_close"]

            _,_,medium = ta.MACD(               #type: ignore
                df[f"{pair}_close"],
                fastperiod = MEDIUM_MULTIPLES * MACD_FAST,
                slowperiod = MEDIUM_MULTIPLES * MACD_SLOW,
                signalperiod = MEDIUM_MULTIPLES * MACD_SIGNAL
            )
            medium = np.array(medium) / df[f"{pair}_close"]

            _,_,slow = ta.MACD(                 #type: ignore
                df[f"{pair}_close"],
                fastperiod = LONG_MULTIPLES * MACD_FAST,
                slowperiod = LONG_MULTIPLES * MACD_SLOW,
                signalperiod = LONG_MULTIPLES * MACD_SIGNAL
            )
            slow = slow / df[f"{pair}_close"]

            add_df = pd.DataFrame(index=df.index,data={
                f"{pair}_FAST_MACD" : fast,
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

                if pair != BASE_PAIR_NAME:
                    df = df.drop([f"{pair}_{cycle}_MACD"],axis=1)

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

    def calculate_entry_signal_for_all_pairs(self, analyzed_df:pd.DataFrame):
        """ Calculate entry signal for all pairs

        Args:
            analyzed_df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: processed dataframe
        """
        logger.debug("calculate_signal_for_all_pairs:IN")
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
        logger.debug("caluclate_signal_for_all_pairs:OUT")
        return df.copy()

    def get_score(self,diff_mean,diff_fast,diff):
        score = (diff_mean + diff_fast + diff)/3
        return score

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

        col = f"{BASE_PAIR_NAME}_{MARKET_CYCLE}_MACD"

        if row[col]>0:
            is_bull = True
        elif row[col]<0:
            is_bull = False
        else:
            logger.debug(f"{col} is NaN")
            return [],[]

        long_dict = dict()
        short_dict = dict()
        inv_long_dict = dict()
        inv_short_dict = dict()

        for pair in pairs:
            if (is_bull):
                diff_fast = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE_FAST}_BULL"])
                diff = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE}_BULL"])
                diff_mean = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE}_BULL_MEAN"])
            else:
                diff_fast = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE_FAST}_BEAR"])
                diff = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE}_BEAR"])
                diff_mean = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE}_BEAR_MEAN"])

            if (diff_mean > 0) & (diff > 0) & (diff_fast > 0):
                long_dict[pair] = diff_mean
                inv_long_dict[diff_mean] = pair
            elif (diff_mean < 0) & (diff < 0) & (diff_fast < 0):
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

    def get_signal_from_lists(
        self,current_pair:str,long_pairs:List[str],short_pairs:List[str]):
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
            print(f"LONG {current_pair}")
            long_signal = 1

        if current_pair in short_pairs:
            short_signal = 1
            print(f"SHORT {current_pair}")

        return long_signal, short_signal

    def populate_shared_analyzed_dataframe(self):
        """ Produce the shared analyzed dataframe
        Returns:
            DataFrame: dataframe
        """
        logger.debug("populate_shared_analyzed_dataframe:IN")
        pairs = self.dp.current_whitelist()
        dataframe:Optional[pd.DataFrame] = None

        for pair in self.dp.current_whitelist():
            informative:pd.DataFrame= self.dp.get_pair_dataframe(
                pair = pair,
                timeframe = self.timeframe
            )
            informative = informative[["date","close"]]
            informative = informative.rename(columns = {"close":f"{pair}_close"})
            if dataframe is None:
                dataframe = informative
            else:
                dataframe = dataframe.merge(informative,on="date")

        if dataframe is None:
            logger.info("calculated_shared_analyzed_dataframe: return EMPTY")
            return pd.DataFrame()

        dataframe = self._calculate_macd(dataframe,pairs)
        dataframe = self._calculate_macd_diff_with_base_pair(dataframe,pairs)
        dataframe = self._calculate_macd_diff_mean(dataframe,pairs)

        logger.debug("populate_shared_analyzed_dataframe:OUT")
        return dataframe

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
        logger.debug("populate_indicator:IN")
        logger.debug("populate_indicator:OUT")
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        logger.debug("populate_entry_trend:IN")
        logger.info(f"populate_entry_trend: {metadata['pair']}")

        shared_dataframe,_ = self.get_shared_analyzed_dataframe(self.timeframe,True)
        if (shared_dataframe is None):
            shared_dataframe = self.populate_shared_analyzed_dataframe()
            self._set_cached_shared_data_df(self.timeframe,shared_dataframe,self.config['candle_type_def'])

        latest = dataframe.iloc[-1]["date"]
        if (shared_dataframe.iloc[-1]["date"] != dataframe.iloc[-1]["date"]):
            raise ValueError("last candle not matching")

        start_date = dataframe.iloc[0]["date"]
        end_date = dataframe.iloc[-1]["date"]

        time_range = TimeRange(starttype=start_date,stoptype=end_date)
        shared_dataframe = trim_dataframe(shared_dataframe,time_range)

        if (latest != self._cached_signal_time) | (self._cached_signal_data is None):
            df = self.calculate_entry_signal_for_all_pairs(shared_dataframe)
            self._cached_signal_data = df
            self._cached_signal_time = latest
        else:
            df = self._cached_signal_data

        pair = metadata["pair"]
        dataframe["enter_long"] = df[f"{pair}_LONG"]    #type: ignore
        dataframe["enter_short"] = df[f"{pair}_SHORT"]  #type: ignore
        logger.debug(f"n_long: {dataframe['enter_long'].sum()}, n_short: {dataframe['enter_short'].sum()}")
        logger.debug("populate_entry_trend:OUT")
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
        self.process_at_loop_start(None)
        logger.debug("bot_loop_start: OUT")

    def bot_start(self):
        """Thing to do at bot start
        """
        logger.debug("bot_start: IN")
        logger.debug("bot_start: OUT")

    def get_shared_analyzed_row(self, date:datetime):
        """ Get shared analyzed data for a specific date

        Args:
            date (datetime): the date to get info

        Returns:
            pd.Series: the analyzed data of the date
        """
        dataframe, _ = self.get_shared_analyzed_dataframe(self.timeframe,False)
        latest_candle = dataframe[dataframe["date"] == date]

        if len(latest_candle) > 0:
            return latest_candle.iloc[0]
        else:
            return None

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
        open_pairs = [trade.pair for trade in Trade.get_open_trades()]
        latest_time = current_time - timedelta(minutes=TRADING_TIMEFRAME_IN_MIN)
        latest_candle = self.get_shared_analyzed_row(latest_time)

        if latest_candle is None:
            raise ValueError("Could not find the candle, weird")

        long_pairs, short_pairs = self.arbitrate_both_sides(
            row = latest_candle,
            pairs=self.dp.current_whitelist()
        )

        if self._cached_stake_time == latest_candle["date"]:
            long_stake = self._cached_long_stake
            short_stake = self._cached_short_stake
        else:
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

        #Update adjust position dictionary
        latest = latest_candle["date"]
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
        total_stake = self.wallets.get_total_stake_amount()
        stake_usage = round(current_stake/total_stake if total_stake > 0 else 0,3)
        total_stake_value = self.wallets.get_total_stake_amount()
        logger.info(f"{current_time} | Long stake pct: {long_stake_ratio}| \
            Used stake pct: {stake_usage}| Total stake value: {total_stake_value}")

        logger.debug("process_at_loop_start:OUT")

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
        latest_time = current_time - timedelta(minutes=TRADING_TIMEFRAME_IN_MIN)
        adjust_position_dict: Optional[Tuple] = self._cached_adjust_position[pair] \
            if pair in self._cached_adjust_position.keys() else None

        if (adjust_position_dict is None):
            return None

        if latest_time not in adjust_position_dict.keys(): #type:ignore
            return None

        if adjust_position_dict[latest_time] == -1:  #type:ignore
            return None

        (side,amount) = adjust_position_dict[latest_time]  #type:ignore
        adjust_position_dict[latest_time] = -1 #type:ignore

        if (trade.trade_direction == side):
            # logger.info(f"Increase {pair} by {amount}")
            result_amount = amount
        else:
            # logger.info(f"Decrease {pair} by {amount}")
            if amount < trade.stake_amount:
                result_amount = -amount
            else:
                result_amount = trade.stake_amount

        return result_amount

    def get_shared_analyzed_dataframe(self, timeframe:str, is_trimmed:bool=True) -> Tuple[DataFrame, datetime]:
        """ Get the analyzed dataframe that store analysis info of all pairs.
            Changed the way freqtrade work to optimize the memory.

        Args:
            timeframe (str): trading timeframe
            is_trimmed (bool, optional): we call trimmed for backtesting. Defaults to True.

        Returns:
            Tuple[DataFrame, datetime]: dataframe and the cached time
        """
        logger.debug("get_shared_analyzed_dataframe:IN")
        key = (timeframe, self.config.get('candle_type_def', CandleType.SPOT))
        if key in self._cached_shared_data:
            df, date = self._cached_shared_data[key]
            if is_trimmed:
                if self.dp._slice_index:
                    max_index = self.dp._slice_index
                    df = df.iloc[max(0, max_index - MAX_DATAFRAME_CANDLES):max_index]

            logger.debug("get_shared_analyzed_dataframe:OUT")
            return df, date
        else:
            logger.debug("return empty df")
            logger.debug("get_shared_analyzed_dataframe:OUT")
            return (None, datetime.fromtimestamp(0, tz=timezone.utc))

    def _set_cached_shared_data_df(
        self,
        timeframe: str,
        dataframe: DataFrame,
        candle_type: CandleType
    ) -> None:
        """
        Store cached Dataframe.
        :param pair: pair to get the data for
        :param timeframe: Timeframe to get data for
        :param dataframe: analyzed dataframe
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        logger.debug("_set_cached_shared_data:IN")
        key = (timeframe, candle_type)
        self._cached_shared_data[key] = (
            dataframe, datetime.now(timezone.utc))
        logger.debug("_set_cached_shared_data:OUT")

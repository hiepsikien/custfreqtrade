
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import sys
import os
import psutil
from freqtrade.constants import Config
from freqtrade.data.dataprovider import MAX_DATAFRAME_CANDLES
from freqtrade.enums.runmode import RunMode
sys.path.append("/home/andy/CryptoTradingPlatform/freqtrade")

from datetime import datetime
import numpy as np  # noqaclear
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, List, Tuple, Dict
from freqtrade.strategy import IStrategy
from freqtrade.enums import CandleType
from datetime import timezone, timedelta
from freqtrade.data.dataprovider import MAX_DATAFRAME_CANDLES
from freqtrade.data.history import load_pair_history

# --------------------------------
# Add your lib to import here
# import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib as ta
import logging
from freqtrade.persistence.trade_model import Trade
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# This class is a sample. Feel free to customize it.

# Market related parameters
VERSION = "1.1.0"
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
#Trading related parameters
TIMEFRAME = "4h"
PAIR_NUMBER = 5
HOLDING_PERIOD_IN_TIMEFRAMES = 3 * 6
INVEST_ROUNDS = HOLDING_PERIOD_IN_TIMEFRAMES / 2
LEVERAGE_RATIO = 2.0
TAKE_PROFIT_RATE = 0.5
STOP_LOSS_RATE = -0.25
STARTUP_CANDLE_COUNT = 30 * 4
HISTORIC_CANDLE_COUNT = 365 * 4

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
        f"{HOLDING_PERIOD_IN_TIMEFRAMES * TIMEFRAME_IN_MIN[TIMEFRAME]}":-1
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
        self.custom_stake_amount_dict: Dict[datetime,Tuple[float,float]] = dict()
        self.custom_adjust_position_amount_dict = dict()
        self.custom_shared_analyzed_data_dict: Dict[Tuple[str,CandleType], Tuple[DataFrame, datetime]] = {}
        self.custom_cached_shared_analyzed_data_time: Optional[str] = None
        self.custom_trade_signal_dict:Dict[datetime,Tuple[List[str],List[str]]] = dict()

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
            # _,_,fast = ta.MACD(                 #type: ignore
            #     df[f"{pair}_close"],
            #     # fastperiod = SHORT_MULTIPLES * MACD_FAST,
            #     slowperiod = SHORT_MULTIPLES * MACD_SLOW,
            #     signalperiod = SHORT_MULTIPLES * MACD_SIGNAL
            # )
            # fast = fast/ df[f"{pair}_close"]

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
                # f"{pair}_FAST_MACD" : fast,
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
        long_pairs, short_pairs, _,_ = self.arbitrate_both_sides(
            row = row,
            pairs = self.dp.current_whitelist()
        )
        self.custom_trade_signal_dict[row["date"]] = (long_pairs,short_pairs)

    def get_new_shared_analyzed_dataframe(self):
        """ Produce the shared analyzed dataframe
        Returns:
            DataFrame: dataframe
        """
        logger.debug("get_new_shared_analyzed_dataframe:IN")
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
                del informative

        if dataframe is None:
            logger.info("calculated_shared_analyzed_dataframe: return EMPTY")
            return pd.DataFrame()

        dataframe = self._calculate_macd(dataframe,pairs)
        dataframe = self._calculate_macd_diff_with_base_pair(dataframe,pairs)
        dataframe = self._calculate_macd_diff_mean(dataframe,pairs)

        logger.debug("get_new_shared_analyzed_dataframe:OUT")
        return dataframe

    def update_analyzed_dataframe(self):
        """ Update the shared analyzed dataframe

        Args:
            end_date (_type_): the last candle date

        Returns:
            _type_: _description_
        """
        logger.debug("update_analyzed_dataframe:IN")
        old_df,_ = self.get_shared_analyzed_dataframe(self.timeframe)

        new_df= None
        pairs = self.dp.current_whitelist()
        for pair in pairs:
            informative:pd.DataFrame= self.dp.get_pair_dataframe(
                pair = pair,
                timeframe = self.timeframe
            )
            informative = informative[["date","close"]]
            informative = informative.rename(columns = {"close":f"{pair}_close"})
            if new_df is None:
                new_df = informative
            else:
                new_df = new_df.merge(informative,on="date")
                del informative

        if new_df is None:
            logger.info("calculated_shared_analyzed_dataframe: return EMPTY")
            return pd.DataFrame()

        new_df = self._calculate_macd(new_df,pairs)
        new_df = self._calculate_macd_diff_with_base_pair(new_df,pairs)
        new_df = self._calculate_macd_diff_mean(new_df,pairs)

        shared_df: pd.DataFrame = pd.concat([old_df,new_df])
        shared_df = shared_df.drop_duplicates(subset=['date'])
        logger.debug(f"old:{len(old_df)} new:{len(new_df)} merged:{len(shared_df)}")
        del old_df, new_df
        self._set_cached_shared_data_df(self.timeframe,shared_df,self.config['candle_type_def'])
        logger.debug("update_analyzed_dataframe:OUT")
        return shared_df

    def calculate_entry_signal_for_all_pairs(self, analyzed_df:pd.DataFrame):
        """ Calculate entry signal for all pairs

        Args:
            analyzed_df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: processed dataframe
        """
        logger.debug("calculate_signal_for_all_pairs:IN")
        analyzed_df.apply(self.foo,axis=1)

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
            return [],[], dict(), dict()

        long_dict: Dict[str,float] = dict()
        short_dict: Dict[str,float] = dict()
        inv_long_dict: Dict[float,str] = dict()
        inv_short_dict: Dict[float,str] = dict()

        for pair in pairs:
            if (is_bull):
                # diff_fast = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE_FAST}_BULL"])
                diff = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE}_BULL"])
                diff_mean = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE}_BULL_MEAN"])
            else:
                # diff_fast = float(row[f"{pair}_DIFF_{ALTCOIN_CYCLE_FAST}_BEAR"])
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

        return long_pairs, short_pairs, long_dict, short_dict

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
        latest_date = dataframe.iloc[-1]["date"]

        if self.custom_cached_shared_analyzed_data_time is None:
            df = self.load_historic_analyzed_dataframe()
            logger.debug(f"len_historic: {len(df)}")
            self._set_cached_shared_data_df(
                timeframe = self.timeframe,
                candle_type=self.config['candle_type_def'],
                dataframe=df)

        if (latest_date != self.custom_cached_shared_analyzed_data_time):
            self.update_analyzed_dataframe()

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

        start_date = dataframe.iloc[0]["date"]
        end_date = dataframe.iloc[-1]["date"]

        if (end_date not in self.custom_trade_signal_dict.keys()):
            shared_dataframe,_ = self.get_shared_analyzed_dataframe(self.timeframe)
            logger.debug(f"Dataframe date = {start_date}:{end_date}")
            logger.debug(f"Shared_dataframe date = {shared_dataframe.iloc[0]['date']}\
                :{shared_dataframe.iloc[-1]['date']}")

            if (shared_dataframe is None) or shared_dataframe.iloc[-1]["date"] != end_date:
                raise ValueError("last candle not matching")
            self.calculate_entry_signal_for_all_pairs(shared_dataframe)

        dataframe["enter_long"] = dataframe.apply(
            lambda row: self._get_signal_long(row["date"],metadata["pair"]),
            axis=1
        )
        dataframe["enter_short"] = dataframe.apply(
            lambda row: self._get_signal_short(row["date"],metadata["pair"]),
            axis=1
        )
        logger.debug("populate_entry_trend:OUT")
        return dataframe.copy()

    def _get_signal_long(self,date:datetime,pair:str):
        long_pairs,_ = self.custom_trade_signal_dict[date]
        signal = 1 if pair in long_pairs else 0
        return signal

    def _get_signal_short(self,date:datetime,pair:str):
        _, short_pairs = self.custom_trade_signal_dict[date]
        signal = 1 if pair in short_pairs else 0
        return signal

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        return dataframe

    def load_historic_analyzed_dataframe(self):
        """Get historic analyzed dataframe for all pair

        Returns:
            pd.Dataframe: results
        """
        logger.debug("get_historic_analyzed_dataframe:IN")
        pairs = self.dp.current_whitelist()
        dataframe:Optional[pd.DataFrame] = None
        for pair in self.dp.current_whitelist():
            informative:pd.DataFrame= load_pair_history(
                pair=pair,
                timeframe=self.config['timeframe'],
                datadir=self.config['datadir'],
                startup_candles=HISTORIC_CANDLE_COUNT,
                data_format=self.config.get('dataformat_ohlcv', 'json'),
                candle_type=self.config['candle_type_def']
            )
            informative = informative[["date","close"]]
            informative = informative.rename(columns = {"close":f"{pair}_close"})
            if dataframe is None:
                dataframe = informative
            else:
                dataframe = dataframe.merge(informative,on="date")
                del informative

        if dataframe is None:
            logger.info("calculated_shared_analyzed_dataframe: return EMPTY")
            return pd.DataFrame()

        dataframe = self._calculate_macd(dataframe,pairs)
        dataframe = self._calculate_macd_diff_with_base_pair(dataframe,pairs)
        dataframe = self._calculate_macd_diff_mean(dataframe,pairs)

        logger.debug("get_historic_analyzed_dataframe:OUT")
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
        self.process_at_loop_start(None)    #type:ignore
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
        logger.debug("custom_stake_amount: IN")
        open_pairs = [trade.pair for trade in Trade.get_open_trades()]

        latest_time = current_time - timedelta(minutes=TIMEFRAME_IN_MIN[TIMEFRAME])

        if latest_time in self.custom_stake_amount_dict.keys():
            long_stake, short_stake = self.custom_stake_amount_dict[latest_time]
        else:

            if latest_time not in self.custom_trade_signal_dict.keys():
                return 0

            long_pairs, short_pairs = self.custom_trade_signal_dict[latest_time]
            logger.debug(f"Discovered LONG:")
            logger.debug([pair for pair in long_pairs])
            logger.debug(f"Discovered SHORT:")
            logger.debug([pair for pair in short_pairs])

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

            self.custom_stake_amount_dict[latest_time] = (long_stake,short_stake)

            #Update adjust position dictionary
            for pair in long_pairs:
                if pair in open_pairs:
                    if pair not in self.custom_adjust_position_amount_dict.keys():
                        self.custom_adjust_position_amount_dict[pair] = dict()
                    if latest_time not in self.custom_adjust_position_amount_dict[pair].keys():
                        self.custom_adjust_position_amount_dict[pair][latest_time] = ("long",long_stake)

            for pair in short_pairs:
                if pair in open_pairs:
                    if pair not in self.custom_adjust_position_amount_dict.keys():
                        self.custom_adjust_position_amount_dict[pair] = dict()
                    if latest_time not in self.custom_adjust_position_amount_dict[pair].keys():
                            self.custom_adjust_position_amount_dict[pair][latest_time] = ("short",short_stake)

        if side == "long":
            logger.debug(f"Long {pair} by {long_stake}")
            logger.debug("custom_stake_amount: OUT")
            return long_stake   #type: ignore
        else:
            logger.debug(f"Short {pair} by {short_stake}")
            logger.debug("custom_stake_amount: OUT")
            return short_stake  #type: ignore

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
        #Print key info
        long_sum, short_sum,_,_ = self.get_open_trades_info()
        current_stake = long_sum + short_sum
        long_stake_ratio = round(long_sum/(long_sum+short_sum) if (long_sum+short_sum)>0 else 0,2)
        total_stake = self.wallets.get_total_stake_amount()     #type: ignore
        stake_usage = round(current_stake/total_stake if total_stake > 0 else 0,3)
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss/1_000_000)
        logger.info(f"{current_time}| Long pct: {long_stake_ratio}| Used pct: "\
            f"{stake_usage}| Open trades: | Total stake: {total_stake}USDT| Mem: {mem_usage}M")

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
        long_pairs, short_pairs = self.custom_trade_signal_dict[latest_time]
        logger.info("TO LONG:{}".format(long_pairs))
        logger.info("TO SHORT:{}".format(short_pairs))
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
        logger.debug("adjust_trade_position:IN")
        pair: str = trade.pair
        latest_time = current_time - timedelta(minutes=TIMEFRAME_IN_MIN[TIMEFRAME])
        adjust_position_dict: Optional[Tuple] = self.custom_adjust_position_amount_dict[pair] \
            if pair in self.custom_adjust_position_amount_dict.keys() else None

        if (adjust_position_dict is None):
            logger.debug(f"{current_time} {pair} null dict")
            logger.debug("adjust_trade_position:OUT")
            return None

        if latest_time not in adjust_position_dict.keys(): #type:ignore
            logger.debug(f"{current_time} time key not found in dict")
            logger.debug("adjust_trade_position:OUT")
            return None

        if adjust_position_dict[latest_time] == -1:  #type:ignore
            logger.debug(f"{trade.pair} adjusted")
            logger.debug("adjust_trade_position:OUT")
            return None

        (side,amount) = adjust_position_dict[latest_time]  #type:ignore
        adjust_position_dict[latest_time] = -1 #type:ignore

        if (trade.trade_direction == side):
            logger.info(f"Increase {side} position of {pair} by {amount}")
            result_amount = amount
        else:
            logger.info(f"Decrease {side} position of {pair} by {amount}")
            if amount < trade.stake_amount:
                result_amount = -amount
            else:
                result_amount = trade.stake_amount

        logger.debug("adjust_trade_position:OUT")
        return result_amount

    def get_shared_analyzed_dataframe(self, timeframe:str) -> Tuple[DataFrame, datetime]:
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
        if key in self.custom_shared_analyzed_data_dict:
            df, date = self.custom_shared_analyzed_data_dict[key]
            logger.debug("get_shared_analyzed_dataframe:OUT")
            return df, date
        else:
            logger.debug("return empty df")
            logger.debug("get_shared_analyzed_dataframe:OUT")
            return (None, datetime.fromtimestamp(0, tz=timezone.utc))   #type:ignore

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
        self.custom_shared_analyzed_data_dict[key] = (
            dataframe, datetime.now(timezone.utc))
        self.custom_cached_shared_analyzed_data_time = dataframe.iloc[-1]["date"]
        logger.debug("_set_cached_shared_data:OUT")

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

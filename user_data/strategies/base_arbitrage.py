
# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import os
import psutil
from freqtrade.constants import Config
from freqtrade.enums.runmode import RunMode
# sys.path.append("/home/andy/CryptoTradingPlatform/freqtrade")

from datetime import datetime
import numpy as np  # noqaclear
import pandas as pd  # noqa
from pandas import DataFrame
from typing import Optional, List, Tuple, Dict
from freqtrade.strategy import IStrategy
from freqtrade.enums import CandleType
from datetime import timezone, timedelta
from freqtrade.data.history import load_pair_history
from freqtrade.configuration.timerange import TimeRange
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
BASE_PAIR_NAME = "BTC/USDT:USDT"
STARTUP_CANDLE_COUNT = 3 * 4 * 24
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
PERIOD_NUMBER_YEARLY = {
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365
}
MAX_CACHED_LENGTH = 20_000
SAVE_PATH = "user_data/db/json"

class BaseArbitrage(IStrategy):
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
     # Can this strategy go short?
    can_short: bool = True
    timeframe = "15m"

    # Custom class variable not changing much
    custom_long_multiple = 9
    custom_medium_multiple = 3
    custom_short_multiple = 1
    custom_market_cycle = "SLOW"
    custom_altcoin_cycle = "MEDIUM"
    custom_market_cycle_list = ["MEDIUM","SLOW"]

    # Custom class variable may vary
    # custom_allow_buy_more:bool=True
    custom_rebalance_long_shot:bool=True
    custom_pair_number:int = 10
    custom_leverage_ratio:float = 2.0
    custom_take_profit_rate:float = 0.1
    custom_stop_loss_rate:float= -0.1
    custom_historic_preloaded_days:int = 90
    custom_holding_period:int = 3 * 6
    custom_invest_rounds:int = 3

    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3
    minimal_roi = {
        "0": custom_take_profit_rate * custom_leverage_ratio,
        f"{custom_holding_period * TIMEFRAME_IN_MIN[timeframe]}":-1
    }
    stoploss = custom_stop_loss_rate * custom_leverage_ratio
    trailing_stop = False
    process_only_new_candles = True
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
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
        self.custom_long_short_amount_dict: Dict[datetime,Tuple[float,float]] = dict()
        self.custom_adjust_amount_dict: Dict[str,Dict[datetime,Tuple[str,float]]]= dict()
        self.custom_cached_data_dict: Dict[Tuple[str,CandleType], Tuple[DataFrame, datetime]] = dict()
        self.custom_cached_data_time: Optional[datetime] = None
        self.custom_entry_signal_dict:Dict[datetime,Tuple[List[str],List[str]]] = dict()
        self.custom_missing_dict:Dict[datetime,List[str]] = dict()

    # def save_data(self):
    #     with open(self.custom_json_path,"w") as f:
    #         json.dump(self.custom_adjust_amount_dict,f)

    # def load_data(self):
    #     if not os.path.exists(self.custom_json_path):
    #         return dict()
    #     with open(self.custom_json_path) as f:
    #         return json.load(f)

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
        # logger.debug("information_pairs:IN")
        # get access to all pairs available in whitelist.
        # pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        # informative_pairs = [(pair, self.timeframe) for pair in pairs]
        # logger.debug("information_pairs:OUT")
        return []

    def analyze_dataframe(self, df:pd.DataFrame,pairs:List[str]):
        """ Analyze dataframe

        Args:
            df (pd.DataFrame): input dataframe
            pairs (List[str]): list of informative pairs

        Returns:
            pd:DataFrame: process dataframe
        """
        logger.debug("analyze_dataframe: IN")
        logger.debug("...calculating MACD ...")
        #Calculate MACD
        for pair in pairs:
            # logger.debug(pair)
            # _,_,fast = ta.MACD(                 #type: ignore
            #     df[f"{pair}_close"],
            #     # fastperiod = SHORT_MULTIPLES * MACD_FAST,
            #     slowperiod = SHORT_MULTIPLES * MACD_SLOW,
            #     signalperiod = SHORT_MULTIPLES * MACD_SIGNAL
            # )
            # fast = fast/ df[f"{pair}_close"]
            col_name = f"{pair}_close"
            _,_,medium = ta.MACD(               #type: ignore
                df[col_name],
                fastperiod = self.custom_medium_multiple * MACD_FAST,
                slowperiod = self.custom_medium_multiple * MACD_SLOW,
                signalperiod = self.custom_medium_multiple * MACD_SIGNAL
            )
            medium = np.array(medium) / df[col_name]

            _,_,slow = ta.MACD(                 #type: ignore
                df[col_name],
                fastperiod = self.custom_long_multiple * MACD_FAST,
                slowperiod = self.custom_long_multiple * MACD_SLOW,
                signalperiod = self.custom_long_multiple * MACD_SIGNAL
            )
            slow = slow / df[col_name]

            add_df = pd.DataFrame(index=df.index,data={
                # f"{pair}_FAST_MACD" : fast,
                f"{pair}_MEDIUM_MACD" : medium,
                f"{pair}_SLOW_MACD" : slow
            })
            df = df.drop([col_name],axis=1)
            df = pd.concat([df,add_df],axis=1)
            del add_df

        # df.fillna(method="ffill", inplace = True)

        #Calculate diff of macd between the coin and base coin
        logger.debug("...calculating MACD diff and diff mean...")
        for pair in pairs:
            # logger.debug(pair)
            for cycle in self.custom_market_cycle_list:
                diff = df[f"{pair}_{cycle}_MACD"] - df[f"{BASE_PAIR_NAME}_{cycle}_MACD"]
                diff.name = f"{pair}_DIFF_{cycle}"

                diff_bear = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]<0][f"{pair}_{cycle}_MACD"] \
                    - df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]<0][f"{BASE_PAIR_NAME}_{cycle}_MACD"]
                diff_bear.name = f"{pair}_DIFF_{cycle}_BEAR"

                diff_bull = df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]>0][f"{pair}_{cycle}_MACD"] \
                - df[df[f"{BASE_PAIR_NAME}_SLOW_MACD"]>0][f"{BASE_PAIR_NAME}_{cycle}_MACD"]
                diff_bull.name = f"{pair}_DIFF_{cycle}_BULL"

                diff_bear_mean = diff_bear.expanding().mean()
                diff_bear_mean.name = f"{pair}_DIFF_{cycle}_BEAR_MEAN"

                diff_bull_mean = diff_bull.expanding().mean()
                diff_bull_mean.name = f"{pair}_DIFF_{cycle}_BULL_MEAN"

                if pair != BASE_PAIR_NAME:
                    df = df.drop([f"{pair}_{cycle}_MACD"],axis=1)

                df = pd.concat([df,diff,diff_bear,diff_bull,diff_bear_mean,diff_bull_mean],axis=1)
                del diff, diff_bear,diff_bull,diff_bear_mean,diff_bull_mean

        logger.debug("analyze_dataframe: OUT")

        return df

    def update_analyzed_dataframe(self,latest_time):
        """ Update the shared analyzed dataframe
        Returns:
            _type_: _description_
        """
        logger.debug("update_analyzed_dataframe:IN")
        old_df,_ = self.get_shared_analyzed_dataframe(self.timeframe)

        new_df: Optional[pd.DataFrame] = None
        pairs = self.dp.current_whitelist()
        missing_pairs = []

        for pair in pairs:
            pair_df:pd.DataFrame= self.dp.get_pair_dataframe(
                pair = pair,
                timeframe=self.timeframe,
                candle_type=self.config["candle_type_def"]
            )
            pair_df = pair_df[["date","close"]]
            pair_df = pair_df.rename(columns = {"close":f"{pair}_close"})

            if pair_df.iloc[-1]["date"] < latest_time:
                missing_pairs.append(pair)

            if new_df is None:
                new_df = pair_df
            else:
                new_df = new_df.merge(pair_df,on="date",how="outer")

        self.custom_missing_dict[latest_time] = missing_pairs
        # new_df.fillna(method="ffill",inplace=True)
        new_df = self.analyze_dataframe(new_df,pairs)

        shared_df: pd.DataFrame = pd.concat([old_df,new_df])
        shared_df = shared_df.drop_duplicates(subset=['date'])
        del old_df, new_df

        self._set_cached_shared_data_df(
            timeframe = self.timeframe,
            dataframe = shared_df,
            candle_type = self.config['candle_type_def'],
            latest_time = latest_time)

        logger.debug("update_analyzed_dataframe:OUT")

        return shared_df, True

    def arbitrage_both_sides(self,
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

        col = f"{BASE_PAIR_NAME}_{self.custom_market_cycle}_MACD"

        if row[col]>0:
            is_bull = True
        elif row[col]<0:
            is_bull = False
        else:
            return [],[], dict(), dict()

        long_dict: Dict[str,float] = dict()
        short_dict: Dict[str,float] = dict()
        inv_long_dict: Dict[float,str] = dict()
        inv_short_dict: Dict[float,str] = dict()

        current_date = row["date"]
        data_missing_pairs = self.custom_missing_dict[current_date] \
            if current_date in self.custom_missing_dict.keys() else []

        for pair in pairs:
            if pair in data_missing_pairs:   #Do not select pair that has missing close data
                continue

            if (is_bull):
                diff = float(row[f"{pair}_DIFF_{self.custom_altcoin_cycle}_BULL"])
                diff_mean = float(row[f"{pair}_DIFF_{self.custom_altcoin_cycle}_BULL_MEAN"])
            else:
                diff = float(row[f"{pair}_DIFF_{self.custom_altcoin_cycle}_BEAR"])
                diff_mean = float(row[f"{pair}_DIFF_{self.custom_altcoin_cycle}_BEAR_MEAN"])

            if (diff_mean > 0) & (diff > 0):
                long_dict[pair] = diff_mean
                inv_long_dict[diff_mean] = pair
            elif (diff_mean < 0) & (diff < 0):
                short_dict[pair] = diff_mean
                inv_short_dict[diff_mean] = pair

        sorted_inv_long_dict = sorted(inv_long_dict,reverse=True)
        long_pairs = [inv_long_dict[sorted_inv_long_dict[i]] \
            for i in range(min(len(long_dict),self.custom_pair_number))]

        sorted_inv_short_dict = sorted(inv_short_dict,reverse=False)
        short_pairs = [inv_short_dict[sorted_inv_short_dict[i]] \
            for i in range(min(len(short_dict),self.custom_pair_number))]

        if long_pairs and not short_pairs:
            short_pairs += [BASE_PAIR_NAME]
            short_dict[BASE_PAIR_NAME] = 0
        elif short_pairs and not long_pairs:
            long_pairs += [BASE_PAIR_NAME]
            long_dict[BASE_PAIR_NAME] = 0

        return long_pairs, short_pairs, long_dict, short_dict

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
        logger.debug(f"for pair {metadata['pair']}")
        latest_date = dataframe.iloc[-1]["date"]

        if self.custom_cached_data_time is None:
            df = self.get_historic_analyzed_dataframe()
            self._set_cached_shared_data_df(
                timeframe = self.timeframe,
                candle_type=self.config['candle_type_def'],
                latest_time= df.iloc[-1]["date"],
                dataframe=df)

        if (latest_date > self.custom_cached_data_time):
            logger.debug(f"update trigger check: {latest_date} <> {self.custom_cached_data_time}")
            self.update_analyzed_dataframe(latest_date)

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

        def _calculate_trade_signal(row):
            long_pairs, short_pairs, \
                _,_ = self.arbitrage_both_sides(
                row = row,
                pairs = self.dp.current_whitelist()
            )
            self.custom_entry_signal_dict[row["date"]] = (long_pairs,short_pairs)

        def _get_signal_long(date:datetime,pair:str):
            long_pairs,_ = self.custom_entry_signal_dict[date]
            signal = 1 if pair in long_pairs else 0
            return signal

        def _get_signal_short(date:datetime,pair:str):
            _, short_pairs = self.custom_entry_signal_dict[date]
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

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        return dataframe

    def get_historic_analyzed_dataframe(self):
        """Get historic analyzed dataframe for all pair

        Returns:
            pd.Dataframe: results
        """
        logger.debug("get_historic_analyzed_dataframe:IN")
        pairs = self.dp.current_whitelist()
        dataframe:Optional[pd.DataFrame] = None

        if self.dp.runmode == RunMode.BACKTEST:
            timerange = TimeRange.parse_timerange(self.config["timerange"])
        else:
            now_time = datetime.now()
            timerange = TimeRange(
                startts= int(datetime.timestamp(now_time)),
                stopts= int(datetime.timestamp(now_time))
            )

        candle_number = int(self.custom_historic_preloaded_days * 24 * 60 \
        / TIMEFRAME_IN_MIN[self.timeframe])

        logger.debug(f"Before substract: {timerange.startdt}:{timerange.stopdt}")
        if candle_number > 0 and timerange:
            timerange.subtract_start(TIMEFRAME_IN_MIN[self.timeframe] * 60 * candle_number)
        logger.debug(f"After substract: {timerange.startdt}:{timerange.stopdt}")
        logger.debug(f"self.config[timeframe] = {self.config['timeframe']}")
        logger.debug(f"self.timeframe = {self.timeframe}")


        for pair in self.dp.current_whitelist():
            df:pd.DataFrame= load_pair_history(
                pair=pair,
                timeframe=self.timeframe,
                datadir=self.config['datadir'],
                data_format=self.config.get('dataformat_ohlcv', 'json'),
                candle_type=self.config['candle_type_def']
            )
            df = df[(df["date"] >= timerange.startdt)&(df["date"]<=timerange.stopdt)]
            df = df[["date","close"]]
            df = df.rename(columns = {"close":f"{pair}_close"})
            if dataframe is None:
                dataframe = df
            else:
                dataframe = dataframe.merge(df,on="date",how="outer")
                del df

        if dataframe is None:
            logger.info("calculated_shared_analyzed_dataframe: return EMPTY")
            return pd.DataFrame()

        dataframe = self.analyze_dataframe(dataframe,pairs)

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
        # logger.debug("custom_stake_amount: IN")
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
            # logger.debug(f"Discovered LONG:")
            # logger.debug([pair for pair in long_pairs])
            # logger.debug(f"Discovered SHORT:")
            # logger.debug([pair for pair in short_pairs])

            invest_budget = min(
                self.wallets.get_total_stake_amount()/self.custom_invest_rounds, #type: ignore
                self.wallets.get_available_stake_amount() #type: ignore
            )
            long_sum, short_sum, _, _ \
                = self.get_open_trades_info()

            if self.custom_rebalance_long_shot:
                long_budget = max(0,0.5 * (short_sum - long_sum + invest_budget))
                short_budget = invest_budget - long_budget
            else:
                long_budget = short_budget = invest_budget/2

            long_number = len(long_pairs)
            short_number = len(short_pairs)

            if min(long_number,short_number) == 0:
                long_number = short_number = 0

            long_pairs = long_pairs[:long_number]
            short_pairs = short_pairs[:short_number]

            long_stake = long_budget/len(long_pairs) if len(long_pairs) > 0 else 0
            short_stake = short_budget/len(short_pairs) if len(short_pairs) > 0 else 0

            self.custom_long_short_amount_dict[latest_time] = (long_stake,short_stake)

            #Update adjust position dictionary
            for pair in long_pairs:
                if pair in open_pairs:
                    if pair not in self.custom_adjust_amount_dict.keys():
                        self.custom_adjust_amount_dict[pair] = dict()
                    if latest_time not in self.custom_adjust_amount_dict[pair].keys():
                        self.custom_adjust_amount_dict[pair][latest_time] = ("long",long_stake)

            for pair in short_pairs:
                if pair in open_pairs:
                    if pair not in self.custom_adjust_amount_dict.keys():
                        self.custom_adjust_amount_dict[pair] = dict()
                    if latest_time not in self.custom_adjust_amount_dict[pair].keys():
                            self.custom_adjust_amount_dict[pair][latest_time] = ("short",short_stake)

        if side == "long":
            # logger.debug(f"Long {pair} by {long_stake}")
            # logger.debug("custom_stake_amount: OUT")
            return long_stake   #type: ignore
        else:
            # logger.debug(f"Short {pair} by {short_stake}")
            # logger.debug("custom_stake_amount: OUT")
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
        return self.custom_leverage_ratio

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
            f"{stake_usage}| Total stake: {total_stake}USDT| Mem: {mem_usage}M")

        #Print current position:
        logger.debug("Current open positions:")
        long_open_trades = []
        short_open_trades = []
        for trade in Trade.get_open_trades():
            if trade.trade_direction == "long":
                long_open_trades.append((trade.pair,f"{trade.stake_amount}{trade.stake_currency}"))
            else:
                short_open_trades.append((trade.pair,f"{trade.stake_amount}{trade.stake_currency}"))

        logger.debug("OPEN LONG: {}".format(long_open_trades))
        logger.debug("OPEN SHORT: {}".format(short_open_trades))

        #Print entry signal
        if current_time:
            latest_time = current_time - timedelta(minutes=TIMEFRAME_IN_MIN[self.timeframe])
            long_pairs, short_pairs = self.custom_entry_signal_dict[latest_time]
            logger.debug("TO LONG:{}".format(long_pairs))
            logger.debug("TO SHORT:{}".format(short_pairs))
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
        latest_time = current_time - timedelta(minutes=TIMEFRAME_IN_MIN[self.timeframe])
        adjust_position_dict: Optional[Dict[datetime,Tuple[str,float]]] = self.custom_adjust_amount_dict[pair] \
            if pair in self.custom_adjust_amount_dict.keys() else None

        if (adjust_position_dict is None):
            logger.debug(f"{current_time} {pair} null dict")
            logger.debug("adjust_trade_position:OUT")
            return None

        if latest_time not in adjust_position_dict.keys(): #type:ignore
            logger.debug(f"{current_time} time key not found in dict")
            logger.debug("adjust_trade_position:OUT")
            return None

        if adjust_position_dict[latest_time][0] == "adjusted":  #type:ignore
            logger.debug(f"{trade.pair} adjusted")
            logger.debug("adjust_trade_position:OUT")
            return None

        (side,amount) = adjust_position_dict[latest_time]  #type:ignore
        adjust_position_dict[latest_time] = ("adjusted",0) #type:ignore

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
        if key in self.custom_cached_data_dict:
            df, date = self.custom_cached_data_dict[key]
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
        latest_time: datetime,
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
        self.custom_cached_data_dict[key] = (
            dataframe.iloc[-min(MAX_CACHED_LENGTH,len(dataframe)):], datetime.now(timezone.utc))
        self.custom_cached_data_time = latest_time
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
        logger.info(f"Enter {side} {pair} of total {round(amount*rate,2)} USDT at rate {rate} USDT")
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
        sell_value = amount*rate
        direction = trade.trade_direction
        profit = amount * (rate - trade.open_rate) if direction == "long" else amount * (trade.open_rate-rate)
        logger.info(f"Exit {trade.trade_direction} {pair} of total {round(sell_value,2)} at rate {rate} USDT, profit {round(profit,2)} USDT as {exit_reason}")

        return True

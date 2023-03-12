import copy
import logging
from typing import List

from freqtrade import constants
from freqtrade.constants import Config
from freqtrade.cutomized_codes.cus_dataprovider import CustomDataProvider
from freqtrade.enums import SignalDirection
from freqtrade.exceptions import DependencyException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.persistence import PairLocks, Trade


logger = logging.getLogger(__name__)

class CusFreqtradeBot(FreqtradeBot):
    """My custom bot
    Args:
        FreqtradeBot (_type_): _description_
    """
    def __init__(self,config: Config) -> None:
        super().__init__(config)
        self.dataprovider = CustomDataProvider(self.config, self.exchange, rpc=self.rpc)

    def _create_trade_for_all_pairs(self, pairs: List[str]) -> int:
        """
        Check the implemented trading strategy for buy signals.

        If the pair triggers the buy signal a new trade record gets created
        and the buy-order opening the trade gets issued towards the exchange.

        :return: True if a trade has been created.
        """
        logger.debug("create_trade_for_all_pairs")

        analyzed_df, _ = self.dataprovider\
            .get_analyzed_dataframe_for_all_pairs(self.strategy.timeframe)

        nowtime = analyzed_df.iloc[-1]['date'] if len(analyzed_df) > 0 else None

        # get_free_open_trades is checked before create_trade is called
        # but it is still used here to prevent opening too many trades within one iteration

        trade_created = 0
        for pair in pairs:
            trade_created+= self.create_trade(pair,analyzed_df,nowtime)

        return trade_created

    def create_trade(self,pair,analyzed_df,nowtime):
        if not self.get_free_open_trades():
            logger.debug(f"Can't open a new trade for {pair}: max number of trades is reached.")
            return False

        # running get_signal on historical data fetched
        (signal, enter_tag) = self.strategy.get_entry_signal(
            pair,
            self.strategy.timeframe,
            analyzed_df
        )

        if signal:
            if self.strategy.is_pair_locked(pair, candle_date=nowtime, side=signal):
                lock = PairLocks.get_pair_longest_lock(pair, nowtime, signal)
                if lock:
                    self.log_once(f"Pair {pair} {lock.side} is locked until "
                                f"{lock.lock_end_time.strftime(constants.DATETIME_PRINT_FORMAT)} "
                                f"due to {lock.reason}.",
                                logger.info)
                else:
                    self.log_once(f"Pair {pair} is currently locked.", logger.info)
                return False

            stake_amount = self.wallets.get_trade_stake_amount(pair, self.edge)

            bid_check_dom = self.config.get('entry_pricing', {}).get('check_depth_of_market', {})
            if ((bid_check_dom.get('enabled', False)) and
                    (bid_check_dom.get('bids_to_ask_delta', 0) > 0)):
                if self._check_depth_of_market(pair, bid_check_dom, side=signal):
                    return self.execute_entry(
                        pair,
                        stake_amount,
                        enter_tag=enter_tag,
                        is_short=(signal == SignalDirection.SHORT)
                    )
                else:
                    return False

            return self.execute_entry(
                pair,
                stake_amount,
                enter_tag=enter_tag,
                is_short=(signal == SignalDirection.SHORT)
            )
        else:
            return False

    def enter_positions(self):
        """
        Tries to execute entry orders for new trades (positions)
        """
        trades_created = 0
        whitelist = copy.deepcopy(self.active_pair_whitelist)

        if not whitelist:
            self.log_once("Active pair whitelist is empty.", logger.info)
            return trades_created
        # Remove pairs for currently opened trades from the whitelist

        for trade in Trade.get_open_trades():
            if trade.pair in whitelist:
                whitelist.remove(trade.pair)
                logger.debug('Ignoring %s in pair whitelist', trade.pair)

        if not whitelist:
            self.log_once("No currency pair in active pair whitelist, "
                          "but checking to exit open trades.", logger.info)
            return trades_created

        if PairLocks.is_global_lock(side='*'):
            # This only checks for total locks (both sides).
            # per-side locks will be evaluated by `is_pair_locked` within create_trade,
            # once the direction for the trade is clear.
            lock = PairLocks.get_pair_longest_lock('*')
            if lock:
                self.log_once(f"Global pairlock active until "
                              f"{lock.lock_end_time.strftime(constants.DATETIME_PRINT_FORMAT)}. "
                              f"Not creating new trades, reason: {lock.reason}.", logger.info)
            else:
                self.log_once("Global pairlock active. Not creating new trades.", logger.info)
            return trades_created
        # Create entity and execute trade for each pair from whitelist

        try:
            trades_created += self._create_trade_for_all_pairs(whitelist)

        except DependencyException as exception:
            logger.warning('Unable to create trade for %s: %s', whitelist, exception)

        if not trades_created:
            logger.debug("Found no enter signals for whitelisted currencies. Trying again...")

        return trades_created

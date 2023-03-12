from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType, RunMode
from freqtrade.exchange import Exchange
from freqtrade.rpc import RPCManager


PairWithTimeframeAndCandleType = Tuple[str, CandleType]
MAX_DATAFRAME_CANDLES = 1000


class CustomDataProvider(DataProvider):
    """ Custom data provider class which enable the integrated dataframe handling

    Args:
        DataProvider (_type_): _description_
    """
    def __init__(self, config: Config,
                 exchange: Optional[Exchange],
                 pairlists=None,
                 rpc: Optional[RPCManager] = None) -> None:

        super().__init__(config, exchange, pairlists, rpc)
        self._cached_super_df: Dict[PairWithTimeframeAndCandleType,
                                    Tuple[DataFrame, datetime]] = {}


    def get_analyzed_dataframe_for_all_pairs(self, timeframe: str) -> Tuple[DataFrame, datetime]:
        """
        Retrieve the analyzed dataframe. Returns the full dataframe in trade mode (live / dry),
        and the last 1000 candles (up to the time evaluated at this moment) in all other modes.
        :param pair: pair to get the data for
        :param timeframe: timeframe to get data for
        :return: Tuple of (Analyzed Dataframe, lastrefreshed) for the requested pair / timeframe
            combination.
            Returns empty dataframe and Epoch 0 (1970-01-01) if no dataframe was cached.
        """
        pair_key = (timeframe, self._config.get('candle_type_def', CandleType.SPOT))
        if pair_key in self.__cached_pairs:
            if self.runmode in (RunMode.DRY_RUN, RunMode.LIVE):
                df, date = self._cached_super_df[pair_key]
            else:
                df, date = self._cached_super_df[pair_key]
                if self.__slice_index is not None:
                    max_index = self.__slice_index
                    df = df.iloc[max(0, max_index - MAX_DATAFRAME_CANDLES):max_index]
            return df, date
        else:
            return (DataFrame(), datetime.fromtimestamp(0, tz=timezone.utc))

    def _set_cached_super_df(
        self,
        timeframe: str,
        dataframe: DataFrame,
        candle_type: CandleType
        ) -> None:
        """
        Store cached Dataframe.
        Using private method as this should never be used by a user
        (but the class is exposed via `self.dp` to the strategy)
        :param pair: pair to get the data for
        :param timeframe: Timeframe to get data for
        :param dataframe: analyzed dataframe
        :param candle_type: Any of the enum CandleType (must match trading mode!)
        """
        pair_key = (timeframe, candle_type)
        self._cached_super_df[pair_key] = (dataframe, datetime.now(timezone.utc))

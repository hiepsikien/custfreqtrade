from base_arbitrage import BaseArbitrage

class AnotherArbitrage(BaseArbitrage):
    timeframe = "15m"

   # Custom class variable not changing much
    custom_long_multiples = 8
    custom_medium_multiples = 3
    custom_short_multiple = 1
    custom_market_cycle = "SLOW"
    custom_altcoin_cycle = "MEDIUM"
    custom_market_cycle_list = ["MEDIUM","SLOW"]

    # Custom class variable may vary
    custom_pair_number = 5
    custom_leverage_ratio = 2.0
    custom_take_profit_rate = 0.1
    custom_stop_loss_rate = -0.1
    custom_historic_preloaded_days  = 7
    custom_holding_period = 24
    custom_invest_rounds = 4

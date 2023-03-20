# import pathlib
# import sys
# sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
from base_arbitrage import BaseArbitrage

class AnotherArbitrage(BaseArbitrage):
    def print_string(self,str):
        print(str)

def main():
    aa = AnotherArbitrage(config={})
    aa.print_string("This is a new arbitrage strategy")
    print(aa.custom_long_multiples)

if __name__ == "__main__":
    main()

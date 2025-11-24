import pandas as pd
import numpy as np

class StockDataProcessor:
    def __init__(self, file_path):
        """
        Initialize the StockDataProcessor with the path to the CSV file.
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.pivot_table = self.data.pivot_table(index="date", values="4. close", columns="ticker")

        self.num_days = self.pivot_table.shape[0]
        self.daily_change = self.pivot_table.pct_change().dropna()

    def get_return(self,):
        """
        Calculate the annualized return based on daily changes.
        The formula used is: (1 + daily_change).prod() ** (252 / num_days)
        where 252 is the average number of trading days in a year.

        Returns:
            
        """
        changes = self.daily_change + 1
        ret_annual = changes.prod() ** (252 / self.num_days)

        return ret_annual
    
    def get_covariance(self,):

        
        mean = self.daily_change.mean()

        relative_change = self.daily_change - mean
        
        covariance = np.dot(relative_change.T, relative_change) * (252 / self.num_days)

        return covariance


    def QUBO_from_Portfolio(self, B, q, t):

        mu = self.get_return()
        sigma = self.get_covariance()
        # Create the QUBO matrix
        Q = np.zeros((len(mu), len(mu)))

        # num_assets 
        num_assets = len(mu)

        R = np.diag(mu)
        S = np.ones((num_assets, num_assets)) - 2 * B * np.diag(np.ones(num_assets)) 

        Q = q * sigma - R + t * S

        return Q
    


# #Example usage:
# processor = StockDataProcessor('all_stock_data.csv')

# # tickers = ["MED", "FDMT", "DCGO"]
# # print(processor.num_days)
# # print(processor.get_return())

# # print(processor.get_covariance())
# print(processor.QUBO_from_Portfolio(0.5, 0.5, 0.5).shape)
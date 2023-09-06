import inspect
from datetime import date, timedelta
from PIMCO.PROPERTY.SQL import SQL
from PIMCO.PROPERTY.TABLES import TABLE, FORMATTING, TABLECONFIG, STYLE, STYLEFNS
from PIMCO.PROPERTY.MULTIINDEX_TABLES import MULTIINDEX_TABLE_SUMMARY as PIVOT
import pandas as pd
import numpy as np
from PIMCO.PROPERTY.MARKET_DATA import BLOOMBER as BBG
import os
from PIMCO.PROPERTY.DATA_TOOL import *
import time
from PIMCO.PROPERTY.REPORTS import REPORTS
from PIMCO.PROPERTY.TABLES_CONGIF import BORDERS, SELECTION
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from six import StringIO
from PIMCO.PROPERTY.REPORTSXL import REPORTSXl
import getpass
from functools import partial


clt = DATA_TOOLClient()
svc = DATA_TOOLService('prod')

class CurrAggregator(object):

    def __init__(self):
        self.accounts = ['14908']
        self.seed_value = 123
        random.seed(self.seed_value)
        self.today = date.today()  # - timedelta(days=5)

    def getEODDates(self):
        today = self.today
        weekday_count = 0
        recent_weekdays = []
        while weekday_count < 2:
            current_date = today - timedelta(days=1)
            if current_date.weekday() < 5:
                recent_weekdays.append(current_date)
                weekday_count += 1
            today = current_date

        lastEOD = recent_weekdays[0].strftime('%Y-%m-%d')
        scndLastEOD = recent_weekdays[1].strftime('%Y-%m-%d')
        return lastEOD, scndLastEOD

    def get_SQL(self, date, acct):
        #MODIFIED SQL CODE FOR PRIVACY
        sql = f'''
        SELECT 
          DATA1.ID as ID,
          DATA1.asof_date AS asof_date,
          DATA2.currency_issue AS currency_issue,
          DATA1.A28 AS PCE,
          DATA1.a12 as currency_exposure,
          DATA1.a279 as no_of_contracts,
          DATA2.QTY_PER_CONTRACT as MULTIPLIER,
          DATA1.strategy_path_3 AS strategy,
          DATA2.comp_sec_type_code,
          DATA2.MATURITY_DATE,
          DATA2.PIMCO_DESC,
          (case when DATA4.comp_sec_type_code = 'OPTN' THEN DATA4.first_coupon_date else DATA2.maturity_date end) as "DELIVERY_DATE",
          DATA1.A2 AS Quantity
        FROM DATA1_OWN.DATA1_STRATEGY DATA1
        LEFT JOIN DATA2_OWN.DATA2_SECURITIES DATA2
          ON DATA2.ID = DATA1.ID
        left join DATA3_own.DATA3_strategy DATA3
            on DATA3.strategy_id = DATA1.STRATEGY_ID
        left join pm_own.pma_sec_attributes_hv DATA4
            on DATA4.ssm_id = DATA1.PARENT_ID
        WHERE DATA1.ACCT_NO IN ({acct})
          AND DATA1.asof_date = TRUNC(DATE '{date}')
          AND DATA1.LT_ID IN (0, 2, 23, 24, 25) AND DATA1.lt_id_2 NOT IN (3, 4, 5)
          AND DATA4.asof_date = TRUNC(DATE '{date}')
          and DATA3.strategy_path_1 = 'NY'
        ORDER BY DATA2.currency_issue
                '''
        df = SQL.query_as_dataframe(sql, conn='PRODUCTION')

        return df

    def get_small_table_Data(self, date, acct):
        df = self.get_SQL(date, acct)

        df = df[['CURRENCY_ISSUE', 'PCE']]
        df = df.groupby(['CURRENCY_ISSUE']).sum().reset_index()
        metrics = df.iloc[:, 1:]
        global metrics_columns
        metrics_columns = metrics.columns.tolist()

        return df

    def currency_filter(self, df, threshold=5):
        global currency_filter
        currency_filter = df[abs(df['PCE']) > threshold]
        currency_filter = currency_filter['CURRENCY_ISSUE'].to_list()
        return threshold

    def get_diff_daily(self, threshold=5):
        lastEOD = self.getEODDates()[0]
        scndLastEOD = self.getEODDates()[1]

        joined_df = pd.DataFrame()
        for acct in self.accounts:
            dfEOD = self.get_small_table_Data(lastEOD, acct)
            dfScndEOD = self.get_small_table_Data(scndLastEOD, acct)
            totals_df = pd.merge(dfEOD, dfScndEOD, on='CURRENCY_ISSUE', how='inner', suffixes=(' ', '  '))

            for index, l in enumerate(metrics_columns):
                totals_df[l] = totals_df.iloc[:, index + 1] - totals_df.iloc[:, index + 1 + len(metrics_columns)]

            columns = totals_df.columns.tolist()
            first_columns = columns[:1]
            last_columns = columns[-len(metrics_columns):]

            new_columns = first_columns + last_columns + columns[1:-len(metrics_columns)]
            totals_df = totals_df[new_columns]

            self.currency_filter(totals_df, threshold)

            return totals_df

    def Prettify(self, threshold):
        lastEOD = self.getEODDates()[0]
        scndLastEOD = self.getEODDates()[1]

        df = self.get_diff_daily()

        mycfg = TABLECONFIG(NUMERIC_PREC=5, TBL_FONT_SIZE='14px', CELL_PADDING='5px', TBL_BG='#ffffff',
                                  ALT_ROW_BG='#f9f9f9')

        higher_than_five = df.loc[abs(df['PCE']) > self.currency_filter(df, threshold)].index.to_list()

        pt = TABLE(df, cfg=FORMATTING + mycfg) \
            .hdr_bg(color='#bdd9ff') \
            .hdr_fg(color="#1A1F25") \
            .font(start_row=0, end_row=1, start_col=0, end_col=-1, weight="bold") \
            .font(start_row=0, end_row=-1, start_col=0, end_col=len(metrics_columns), weight="bold") \
            .cell_applyfn([STYLEFNS.NEGATIVE_FG('red')], cols=list(range(1, len(df.columns)))) \
            .col_numeric(cols=list(range(1, len(df.columns))), prec=3, addcomma=True) \
            .tbl_border(thickness='1px', color='#e2dede', mechanism=BORDERS.GRID) \
            .col_border(color='#b0b0b0', thickness='1px', cols=[i + 1 for i in range(len(metrics_columns))],
                        mechanism=BORDERS.GRID) \
            .hdr_addmore(
            [('CURRENCY_ISSUE', 1), ('DIFFERENCE', len(metrics_columns)), (str(lastEOD), len(metrics_columns)),
             (str(scndLastEOD), len(metrics_columns))]) \
            .cell_span(start_row=0, end_row=1, start_col=0, end_col=0) \
            .row_bg('yellow', rows=higher_than_five)

        return pt

    def run_job1(self, threshold):
        pt = self.Prettify(threshold)
        return pt

    def getDates(self):
        today = today = self.today
        weekday_count = 0
        recent_weekdays = []
        while weekday_count < 30:  # change to get different date range for line chart
            current_date = today - timedelta(days=1)
            if current_date.weekday() < 5:
                recent_weekdays.append(current_date.strftime('%Y-%m-%d'))
                weekday_count += 1
            today = current_date
        return recent_weekdays

    def getDiffDates(self):
        today = today = self.today
        weekday_count = 0
        recent_weekdays = []
        while weekday_count < 30:  # change to get different date range for bar chart
            current_date = today - timedelta(days=1)
            if current_date.weekday() < 5:
                recent_weekdays.append(current_date.strftime('%Y-%m-%d'))
                weekday_count += 1
            today = current_date
        return recent_weekdays

    def get_graphs_table_filtered_Data(self, date, acct, filterUSD):

        df = self.get_SQL(date, acct)
        df = df[['ASOF_DATE', 'CURRENCY_ISSUE', 'PCE']]
        df['ASOF_DATE'] = pd.to_datetime(df['ASOF_DATE'])

        df = df[df['CURRENCY_ISSUE'].isin(currency_filter)]

        df = df.groupby('CURRENCY_ISSUE').agg({'ASOF_DATE': 'max', 'PCE': 'sum'}).reset_index()

        if filterUSD == True:
            df = df[df['CURRENCY_ISSUE'] != 'USD']

        return df

    def get_graphs_table_Data(self, date, acct, filterUSD):

        df = self.get_SQL(date, acct)
        df = df[['ASOF_DATE', 'CURRENCY_ISSUE', 'PCE']]
        df['ASOF_DATE'] = pd.to_datetime(df['ASOF_DATE'])

        df = df.groupby('CURRENCY_ISSUE').agg({'ASOF_DATE': 'max', 'PCE': 'sum'}).reset_index()

        if filterUSD == True:
            df = df[df['CURRENCY_ISSUE'] != 'USD']

        return df

    def get_pce_ot(self, filtr, filterUSD):
        Dates = self.getDates()

        dfs = []

        if filtr == 'yes':

            for acct in self.accounts:
                for day in tqdm(Dates):
                    df = self.get_graphs_table_filtered_Data(day, acct, filterUSD)
                    dfs.append(df)

                df_concat = pd.concat(dfs)

                df_concat['ASOF_DATE'] = pd.to_datetime(df_concat['ASOF_DATE'])

                df_pivot = df_concat.pivot(index='ASOF_DATE', columns='CURRENCY_ISSUE', values='PCE')
        else:
            for acct in self.accounts:
                for day in tqdm(Dates):
                    df = self.get_graphs_table_Data(day, acct, filterUSD)
                    dfs.append(df)

                df_concat = pd.concat(dfs)

                df_concat['ASOF_DATE'] = pd.to_datetime(df_concat['ASOF_DATE'])

                df_pivot = df_concat.pivot(index='ASOF_DATE', columns='CURRENCY_ISSUE', values='PCE')

        return df_pivot

    def get_pce_diff_ot(self, filtr, filterUSD):
        Dates = self.getDiffDates()

        dfs = []

        joined_df = pd.DataFrame()

        if filtr == 'yes':
            for acct in self.accounts:
                for i, day in tqdm(enumerate(Dates)):
                    if i == 0:
                        pass
                    else:
                        dfEOD = self.get_graphs_table_filtered_Data(Dates[i - 1], acct, filterUSD)
                        dfScndEOD = self.get_graphs_table_filtered_Data(Dates[i], acct, filterUSD)
                        totals_df = pd.merge(dfEOD, dfScndEOD, on='CURRENCY_ISSUE', how='inner',
                                             suffixes=('_EOD', '_DB'))
                        totals_df['PCE_diff'] = totals_df['PCE_EOD'] - totals_df['PCE_DB']
                        totals_df = totals_df[['CURRENCY_ISSUE', 'ASOF_DATE_EOD', 'PCE_diff']]
                        dfs.append(totals_df)

                df_concat = pd.concat(dfs)

                df_concat['ASOF_DATE_EOD'] = pd.to_datetime(df_concat['ASOF_DATE_EOD'])

                df_pivot = df_concat.pivot(index='ASOF_DATE_EOD', columns='CURRENCY_ISSUE', values='PCE_diff')

        else:
            for acct in self.accounts:
                for i, day in tqdm(enumerate(Dates)):
                    if i == 0:
                        pass
                    else:
                        dfEOD = self.get_graphs_table_Data(Dates[i - 1], acct, filterUSD)
                        dfScndEOD = self.get_graphs_table_Data(Dates[i], acct, filterUSD)
                        totals_df = pd.merge(dfEOD, dfScndEOD, on='CURRENCY_ISSUE', how='inner',
                                             suffixes=('_EOD', '_DB'))
                        totals_df['PCE_diff'] = totals_df['PCE_EOD'] - totals_df['PCE_DB']
                        totals_df = totals_df[['CURRENCY_ISSUE', 'ASOF_DATE_EOD', 'PCE_diff']]
                        dfs.append(totals_df)

                df_concat = pd.concat(dfs)

                df_concat['ASOF_DATE_EOD'] = pd.to_datetime(df_concat['ASOF_DATE_EOD'])

                df_pivot = df_concat.pivot(index='ASOF_DATE_EOD', columns='CURRENCY_ISSUE', values='PCE_diff')

        return df_pivot

    def plot(self, df_pivot, title, pce_label):

        min_pce = df_pivot.min().min() - 2
        max_pce = df_pivot.max().max() + 2

        nc = len(df_pivot.columns)

        num_columns = min(len(df_pivot.columns), 5)
        num_rows = (len(df_pivot.columns) + num_columns - 1) // num_columns

        if nc < 2:
            for i, currency in enumerate(df_pivot.columns):
                df_pivot = df_pivot[[currency]]  # change to see individual currencies

                df_pivot.plot(kind='line', figsize=(10, 6), color=plt.cm.rainbow(i / len(df_pivot.columns)))
                plt.xlabel('Time')
                plt.ylabel(pce_label)
                plt.title(title)
                plt.grid(True, linewidth=0.2, axis='y')
                plt.ylim(min_pce, max_pce)
                plt.xticks([df_pivot.index[0], df_pivot.index[-1]])

                num_steps = 15
                y_ticks = plt.MaxNLocator(num_steps)
                plt.gca().yaxis.set_major_locator(y_ticks)

                plt.axhline(0, color='black', linewidth=0.5)

                plt.legend(df_pivot.columns)

                plt.show()

                print('\n\n\n')

        elif nc < 6:
            fig, axes = plt.subplots(nrows=1, ncols=nc, figsize=(15, 3 * num_rows))
            fig.suptitle(title, fontsize=16)
            for i, currency in enumerate(df_pivot.columns):
                axes[i].plot(df_pivot.index, df_pivot[currency], color=plt.cm.rainbow(i / len(df_pivot.columns)))
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel(pce_label)
                axes[i].set_title(f'{currency} {pce_label}, Over Time')
                axes[i].set_ylim(min_pce, max_pce)
                axes[i].set_xticks([df_pivot.index[0], df_pivot.index[-1]])
                axes[i].grid(True, linewidth=0.2, axis='y')

                num_steps = 15
                y_ticks = plt.MaxNLocator(num_steps)
                axes[i].yaxis.set_major_locator(y_ticks)

                axes[i].axhline(0, color='black', linewidth=0.5)

            plt.tight_layout()
            plt.show()
            print('\n\n\n')

        else:
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(15, 3 * num_rows))
            fig.suptitle(title, fontsize=16)
            for i, currency in enumerate(df_pivot.columns):
                row = i // num_columns
                col = i % num_columns
                axes[row, col].plot(df_pivot.index, df_pivot[currency], color=plt.cm.rainbow(i / len(df_pivot.columns)))
                axes[row, col].set_xlabel('Time')
                axes[row, col].set_ylabel(pce_label)
                axes[row, col].set_title(f'{currency} {pce_label}, Over Time')
                axes[row, col].set_ylim(min_pce, max_pce)
                axes[row, col].set_xticks([df_pivot.index[0], df_pivot.index[-1]])
                axes[row, col].grid(True, linewidth=0.2, axis='y')

                num_steps = 15
                y_ticks = plt.MaxNLocator(num_steps)
                axes[row, col].yaxis.set_major_locator(y_ticks)

                axes[row, col].axhline(0, color='black', linewidth=0.5)

            if len(df_pivot.columns) < num_rows * num_columns:
                for i in range(len(df_pivot.columns), num_rows * num_columns):
                    fig.delaxes(axes.flatten()[i])

            plt.tight_layout()

            plt.subplots_adjust(top=0.9)

            plt.show()
            print('\n\n\n')

    def diff_plot(self, df_pivot, title, pce_label):
        min_pce = df_pivot.min().min() - 2
        max_pce = df_pivot.max().max() + 2

        nc = len(df_pivot.columns)

        num_columns = min(len(df_pivot.columns), 5)
        num_rows = (len(df_pivot.columns) + num_columns - 1) // num_columns

        if nc < 2:
            for currency in df_pivot.columns:
                plt.figure(figsize=(10, 6))
                colors = ['red' if price < 0 else 'blue' for price in df_pivot[currency]]
                plt.bar(df_pivot.index, df_pivot[currency], color=colors)
                plt.xlabel('Time')
                plt.ylabel(pce_label)
                plt.title(f'{currency} {pce_label}, Over Time')
                plt.grid(True, linewidth=0.2, axis='y')
                plt.ylim(min_pce, max_pce)
                plt.xticks([df_pivot.index[0], df_pivot.index[-1]])

                num_steps = 15
                y_ticks = plt.MaxNLocator(num_steps)
                plt.gca().yaxis.set_major_locator(y_ticks)

                plt.axhline(0, color='black', linewidth=0.5)

                plt.show()
                print('\n\n\n')

        elif nc < 6:
            fig, axes = plt.subplots(nrows=1, ncols=nc, figsize=(15, 3 * num_rows))
            fig.suptitle(title, fontsize=16)
            for i, currency in enumerate(df_pivot.columns):
                colors = ['red' if price < 0 else 'blue' for price in df_pivot[currency]]
                axes[i].bar(df_pivot.index, df_pivot[currency], color=colors)
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel(pce_label)
                axes[i].set_title(f'{currency} {pce_label}, Over Time')
                axes[i].set_ylim(min_pce, max_pce)
                axes[i].set_xticks([df_pivot.index[0], df_pivot.index[-1]])
                axes[i].grid(True, linewidth=0.2, axis='y')

                num_steps = 15
                y_ticks = plt.MaxNLocator(num_steps)
                axes[i].yaxis.set_major_locator(y_ticks)

            plt.tight_layout()
            plt.show()
            print('\n\n\n')

        else:
            fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(15, 3 * num_rows))
            fig.suptitle(title, fontsize=16)

            for i, currency in enumerate(df_pivot.columns):
                row = i // num_columns
                col = i % num_columns
                colors = ['red' if price < 0 else 'blue' for price in df_pivot[currency]]
                axes[row, col].bar(df_pivot.index, df_pivot[currency], color=colors)
                axes[row, col].set_xlabel('Time')
                axes[row, col].set_ylabel(pce_label)
                axes[row, col].set_title(f'{currency} {pce_label}, Over Time')
                axes[row, col].set_ylim(min_pce, max_pce)
                axes[row, col].set_xticks([df_pivot.index[0], df_pivot.index[-1]])
                axes[row, col].grid(True, linewidth=0.2, axis='y')

                num_steps = 15
                y_ticks = plt.MaxNLocator(num_steps)
                axes[row, col].yaxis.set_major_locator(y_ticks)

            if len(df_pivot.columns) < num_rows * num_columns:
                for i in range(len(df_pivot.columns), num_rows * num_columns):
                    fig.delaxes(axes.flatten()[i])

            plt.tight_layout()
            plt.show()
            print('\n\n\n')

    def drill_down(self, df_pivot, title, pce_label, currency):

        min_pce = df_pivot.min().min() - 2
        max_pce = df_pivot.max().max() + 2

        df_pivot = df_pivot[currency]  # change to see individual currencies

        df_pivot.plot(kind='line', figsize=(10, 6))
        plt.xlabel('Time')
        plt.ylabel(pce_label)
        plt.title(title)
        plt.grid(True, linewidth=0.2, axis='y')
        plt.ylim(min_pce, max_pce)
        plt.xticks([df_pivot.index[0], df_pivot.index[-1]])

        num_steps = 15
        y_ticks = plt.MaxNLocator(num_steps)
        plt.gca().yaxis.set_major_locator(y_ticks)

        plt.axhline(0, color='black', linewidth=0.5)

        plt.legend(df_pivot.columns)

        plt.show()

    def diff_drill_down(self, df_pivot, title, pce_label, currency):

        min_pce = df_pivot.min().min() - 2
        max_pce = df_pivot.max().max() + 2

        df_pivot = df_pivot[currency]  # change to see individual currencies differency

        for currency in df_pivot.columns:
            plt.figure(figsize=(10, 6))
            colors = ['red' if price < 0 else 'blue' for price in df_pivot[currency]]
            plt.bar(df_pivot.index, df_pivot[currency], color=colors)
            plt.xlabel('Time')
            plt.ylabel(pce_label)
            plt.title(f'{currency} {pce_label}, Over Time')
            plt.grid(True, linewidth=0.2, axis='y')
            plt.ylim(min_pce, max_pce)
            plt.xticks([df_pivot.index[0], df_pivot.index[-1]])

            num_steps = 15
            y_ticks = plt.MaxNLocator(num_steps)
            plt.gca().yaxis.set_major_locator(y_ticks)

            plt.axhline(0, color='black', linewidth=0.5)

            plt.show()

    def comparison_drill_down(self, df_pivot, df_diff, title, pce_label, currency):
        df_merged = pd.merge(df_pivot, df_diff, left_index=True, right_index=True, how='outer')

        min_pce = df_merged.min().min() - 2
        max_pce = df_merged.max().max() + 2

        df_pivot = df_pivot[currency]  # change to see individual currencies differency

        for i, currency in enumerate(df_pivot.columns):
            plt.figure(figsize=(10, 6))
            colors = ['red' if price < 0 else 'blue' for price in df_merged[f'{currency}_y']]
            plt.bar(df_merged.index, df_merged[f'{currency}_y'], color=colors)
            plt.plot(df_merged.index, df_merged[f'{currency}_x'], color='orange')
            plt.xlabel('Time')
            plt.ylabel(pce_label)
            plt.title(f'{currency} {pce_label}, Over Time')
            plt.grid(True, linewidth=0.2, axis='y')
            plt.ylim(min_pce, max_pce)
            plt.xticks([df_pivot.index[0], df_pivot.index[-1]])

            num_steps = 15
            y_ticks = plt.MaxNLocator(num_steps)
            plt.gca().yaxis.set_major_locator(y_ticks)

            plt.axhline(0, color='black', linewidth=0.5)

            plt.show()

    def get_big_table_Data(self, date, account, filterPlusFive):

        df = self.get_SQL(date, account)

        lastEOD_AE = datetime.strptime(date, '%Y-%m-%d')
        lastEOD_AE = lastEOD_AE.strftime('%Y/%m/%d')

        securities = df['SSM_ID'].to_list()

        flt1 = DATA_TOOLReportFilter('strategy_path_1', 'In', ['NY'])

        df_holdings_delta = svc.DATA_TOOL(["OPTION_DELTA*", "UNDL_PRICE", 'PRICE'], securities=securities,
                                   level=DATA_TOOLLevel.Account, filters=[flt1], startDate=lastEOD_AE)
        df_holdings_delta['OPTION_DELTA*'] = pd.to_numeric(df_holdings_delta['OPTION_DELTA*'], errors='coerce')
        df_holdings_delta['UNDL_PRICE'] = pd.to_numeric(df_holdings_delta['UNDL_PRICE'], errors='coerce')
        df_holdings_delta['PRICE'] = pd.to_numeric(df_holdings_delta['PRICE'], errors='coerce')
        df_holdings_delta = df_holdings_delta.rename(columns={'SECURITY_ID': 'SSM_ID'})

        df_holdings_delta = df_holdings_delta.fillna(0)
        df_holdings_delta = df_holdings_delta.groupby(by='SSM_ID').mean().reset_index()

        joined_df = pd.merge(df, df_holdings_delta, on='SSM_ID', how='left')

        joined_df['DELTA'] = joined_df['OPTION_DELTA*'] * joined_df['UNDL_PRICE'] * joined_df['MULTIPLIER'] * joined_df[
            'NO_OF_CONTRACTS']

        joined_df = joined_df[
            ['CURRENCY_ISSUE', 'STRATEGY', 'COMP_SEC_TYPE_CODE', 'SSM_ID', 'PCE', 'CURRENCY_EXPOSURE', 'QUANTITY',
             'DELTA', 'OPTION_DELTA*', 'MATURITY_DATE', 'PIMCO_DESC', 'DELIVERY_DATE']]

        if filterPlusFive == True:
            joined_df = joined_df[joined_df['CURRENCY_ISSUE'].isin(currency_filter)]

        return joined_df

    def adjust_lightness(self, color, amount=0.5):
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        rgb = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        return hex_color

    def Prettyfy(self, joined_df):
        numeric_columns = joined_df.select_dtypes(include='number')
        numeric_indices = numeric_columns.columns.tolist()

        hm_colors = ['blue', 'white', 'red']

        mycfg = TABLECONFIG(NUMERIC_PREC=5, TBL_FONT_SIZE='14px', CELL_PADDING='5px')
        pt = TABLE(joined_df, cfg=FORMATTING + mycfg) \
            .hdr_bg(color='#bdd9ff') \
            .hdr_fg(color="#1A1F25") \
            .font(start_row=0, end_row=0, start_col=0, end_col=-1, weight="bold") \
            .font(start_row=0, end_row=-1, start_col=0, end_col=6, weight="bold") \
            .col_numeric(cols=numeric_indices, prec=3, addcomma=True) \
            .tbl_border(thickness='0.5px', color='#c3c4c7', mechanism=BORDERS.GRID) \
            .cell_applyfn([STYLEFNS.NEGATIVE_FG('red')], cols=numeric_indices) \
            .col_border(color='#e1e1e3', thickness='3px', cols=range(len(joined_df.columns))) \
            .col_border(color='black', thickness='3px', cols=[6], mechanism=BORDERS.RIGHT) \
            .col_border(color='black', thickness='3px', cols=[6 + len(values_columns)], mechanism=BORDERS.RIGHT) \
            .col_border(color='black', thickness='3px', cols=[6 + len(values_columns) + len(values_columns)],
                        mechanism=BORDERS.RIGHT) \
            .cell_heatmap(palette=hm_colors, cols=['PCE_diff'])

        dfs = {}

        for category in categories:
            dfs[category] = joined_df[joined_df['CURRENCY_ISSUE'] == category]

        for i, (category, df_category) in enumerate(dfs.items()):

            CURRENCY_ISSUE = df_category

            category_rows = CURRENCY_ISSUE.index.tolist()

            start_indices = CURRENCY_ISSUE.loc[
                CURRENCY_ISSUE['CURRENCY_ISSUE'].shift() != CURRENCY_ISSUE['CURRENCY_ISSUE']].index
            end_indices = CURRENCY_ISSUE.loc[
                CURRENCY_ISSUE['CURRENCY_ISSUE'].shift(-1) != CURRENCY_ISSUE['CURRENCY_ISSUE']].index

            red = random.randint(240, 255)
            green = random.randint(240, 255)
            blue = random.randint(240, 255)
            shade = '#{:02x}{:02x}{:02x}'.format(red, green, blue)

            for i in range(len(start_indices)):
                category = CURRENCY_ISSUE.loc[start_indices[i], 'CURRENCY_ISSUE']
                start_index = start_indices[i]
                end_index = end_indices[i]
                pt.cell_span(start_index, end_index, 0, 0, section=SELECTION.BODY)
                pt.row_bg(shade, rows=category_rows)
                pt.row_border(color=self.adjust_lightness(shade, 4), thickness='3px', rows=[start_index],
                              mechanism=BORDERS.BOTTOM)

            STRATEGY = df_category

            start_indices = STRATEGY.loc[STRATEGY['STRATEGY'].shift() != STRATEGY['STRATEGY']].index
            end_indices = STRATEGY.loc[STRATEGY['STRATEGY'].shift(-1) != STRATEGY['STRATEGY']].index

            for i in range(len(start_indices)):
                category = STRATEGY.loc[start_indices[i], 'STRATEGY']
                start_index = start_indices[i]
                end_index = end_indices[i]
                pt.cell_span(start_index, end_index, 1, 1, section=SELECTION.BODY)
                pt.row_border(color=self.adjust_lightness(shade, 4), thickness='3px', rows=[start_index],
                              mechanism=BORDERS.BOTTOM)

            COMP_SEC_TYPE_CODE = df_category

            start_indices = COMP_SEC_TYPE_CODE.loc[
                COMP_SEC_TYPE_CODE['COMP_SEC_TYPE_CODE'].shift() != COMP_SEC_TYPE_CODE['COMP_SEC_TYPE_CODE']].index
            end_indices = COMP_SEC_TYPE_CODE.loc[
                COMP_SEC_TYPE_CODE['COMP_SEC_TYPE_CODE'].shift(-1) != COMP_SEC_TYPE_CODE['COMP_SEC_TYPE_CODE']].index

            for i in range(len(start_indices)):
                category = COMP_SEC_TYPE_CODE.loc[start_indices[i], 'COMP_SEC_TYPE_CODE']
                start_index = start_indices[i]
                end_index = end_indices[i]
                pt.cell_span(start_index, end_index, 2, 2, section=SELECTION.BODY)
                pt.row_border(color=self.adjust_lightness(shade, 4), thickness='3px', rows=[start_index],
                              mechanism=BORDERS.BOTTOM)

        for i, (category, df_category) in enumerate(dfs.items()):

            highlight_df = joined_df[joined_df['CURRENCY_ISSUE'] == category]
            strat_dummy = highlight_df[highlight_df['SSM_ID'].notna() & highlight_df['SSM_ID'].str.startswith('STR')]
            strat_dummy = strat_dummy['SSM_ID'].tolist()
            for ids in strat_dummy:
                pt.cell_applyfn([STYLEFNS.BG('=', ids, '#3afc61')], cols=['SSM_ID'])

            def condition(df, col_name, col_index, row_index, col_value):
                columns_to_check = ['PCE_last_EOD', 'CURRENCY_EXPOSURE_last_EOD', 'DELTA_last_EOD',
                                    'OPTION_DELTA*_last_EOD', 'QUANTITY_last_EOD']
                condition = True
                for column in columns_to_check:
                    condition = condition & (df[column] == 0)
                condition = condition & (df['QUANTITY_day_before'] != 0)
                df = df[condition]
                if row_index in df.index:
                    return STYLE(background='#3afc61')

            pt.cell_applyfn([partial(condition, highlight_df)], cols=['SSM_ID'])

        pt.cell_applyfn(STYLEFNS.BG('=', 'SUBTOTAL', 'yellow'), cols=['SSM_ID'])
        # wb = pt.to_xlsx("test_multiple.xlsx")
        return pt

    def mergeAndCompute(self, filterPlusFive, filterUSD, threshold, barchart='no'):
        self.get_diff_daily(threshold)
        if len(currency_filter) > 0:
            for account in self.accounts:
                lastEOD = self.getEODDates()[0]
                scndLastEOD = self.getEODDates()[1]
                pt1 = self.get_big_table_Data(lastEOD, account, filterPlusFive)
                pt1 = pt1.groupby(by=['CURRENCY_ISSUE', 'STRATEGY', 'COMP_SEC_TYPE_CODE', 'MATURITY_DATE', 'PIMCO_DESC',
                                      'DELIVERY_DATE', 'SSM_ID']).agg(
                    {'PCE': 'sum', 'CURRENCY_EXPOSURE': 'sum', 'DELTA': 'mean', 'OPTION_DELTA*': 'mean',
                     'QUANTITY': 'sum'}).reset_index()
                if filterUSD == True:
                    pt1 = pt1[pt1['CURRENCY_ISSUE'] != 'USD']

                pt2 = self.get_big_table_Data(scndLastEOD, account, filterPlusFive)
                pt2 = pt2.groupby(by=['CURRENCY_ISSUE', 'STRATEGY', 'COMP_SEC_TYPE_CODE', 'MATURITY_DATE', 'PIMCO_DESC',
                                      'DELIVERY_DATE', 'SSM_ID']).agg(
                    {'PCE': 'sum', 'CURRENCY_EXPOSURE': 'sum', 'DELTA': 'mean', 'OPTION_DELTA*': 'mean',
                     'QUANTITY': 'sum'}).reset_index()
                if filterUSD == True:
                    pt2 = pt2[pt2['CURRENCY_ISSUE'] != 'USD']

                pt3 = pd.merge(pt1, pt2,
                               on=['CURRENCY_ISSUE', 'STRATEGY', 'COMP_SEC_TYPE_CODE', 'MATURITY_DATE', 'PIMCO_DESC',
                                   'DELIVERY_DATE', 'SSM_ID'], how='outer', suffixes=('_last_EOD', '_day_before'))

                pt3[['PCE_last_EOD', 'CURRENCY_EXPOSURE_last_EOD', 'DELTA_last_EOD', 'OPTION_DELTA*_last_EOD',
                     'QUANTITY_last_EOD']] = pt3[
                    ['PCE_last_EOD', 'CURRENCY_EXPOSURE_last_EOD', 'DELTA_last_EOD', 'OPTION_DELTA*_last_EOD',
                     'QUANTITY_last_EOD']].fillna(0)

                subtotals = pt3.groupby('CURRENCY_ISSUE').sum().reset_index()
                subtotals['CURRENCY_ISSUE'] = subtotals['CURRENCY_ISSUE']
                subtotals['STRATEGY'] = 'zzzz'
                subtotals['COMP_SEC_TYPE_CODE'] = 'SUBTOTAL'
                subtotals['MATURITY_DATE'] = 'SUBTOTAL'
                subtotals['PIMCO_DESC'] = 'SUBTOTAL'
                subtotals['DELIVERY_DATE'] = 'SUBTOTAL'
                subtotals['SSM_ID'] = 'SUBTOTAL'

                pt3 = pd.concat([pt3, subtotals]).sort_values('CURRENCY_ISSUE', kind='mergesort').reset_index(drop=True)

                global values_columns
                values_columns = pt1.iloc[:, 7:].columns.tolist()

                for i, column in enumerate(values_columns):
                    column_lastEOD = pt3.iloc[:, 7 + i]
                    column_dayBefore = pt3.iloc[:, 7 + i + len(values_columns)]
                    pt3[column + '_diff'] = column_lastEOD - column_dayBefore

                columns = pt3.columns.tolist()
                first_columns = columns[:7]
                last_columns = columns[-len(values_columns):]

                new_columns = first_columns + last_columns + columns[7:-len(values_columns)]
                pt3 = pt3[new_columns]

                global categories
                categories = pt1['CURRENCY_ISSUE'].unique()

                for val in values_columns:
                    pt3[f'{val}_diff'].fillna(pt3[f'{val}_last_EOD'], inplace=True)

                pt3 = pt3.groupby(['CURRENCY_ISSUE', 'STRATEGY', 'COMP_SEC_TYPE_CODE']).apply(
                    lambda x: x.sort_values('PCE_diff', ascending=False)).reset_index(drop=True)

                pt3.loc[pt3['STRATEGY'] == 'zzzz', 'STRATEGY'] = ''
                pt3.loc[pt3['COMP_SEC_TYPE_CODE'] == 'SUBTOTAL', 'COMP_SEC_TYPE_CODE'] = ''
                pt3.loc[pt3['MATURITY_DATE'] == 'SUBTOTAL', 'MATURITY_DATE'] = ''
                pt3.loc[pt3['PIMCO_DESC'] == 'SUBTOTAL', 'PIMCO_DESC'] = ''
                pt3.loc[pt3['DELIVERY_DATE'] == 'SUBTOTAL', 'DELIVERY_DATE'] = ''

                global pt3c
                pt3c = pt3.columns.to_list()

                if barchart == 'no':
                    return self.Prettyfy(pt3)
                else:
                    return pt3

    def get_columns(self):
        self.mergeAndCompute(filterPlusFive=True, filterUSD=True, threshold=5)
        return pt3c

    def get_SQL_untagged(self, date, acct):
        sql = f'''
        SELECT 
          DATA1.ID as ID,
          DATA1.asof_date AS asof_date,
          DATA2.currency_issue AS currency_issue,
          DATA1.A28 AS PCE,
          DATA1.a12 as currency_exposure,
          DATA1.a279 as no_of_contracts,
          DATA2.QTY_PER_CONTRACT as MULTIPLIER,
          DATA1.strategy_path_3 AS strategy,
          DATA2.comp_sec_type_code,
          DATA2.MATURITY_DATE,
          DATA2.PIMCO_DESC,
          (case when DATA4.comp_sec_type_code = 'OPTN' THEN DATA4.first_coupon_date else DATA2.maturity_date end) as "DELIVERY_DATE",
          DATA1.A2 AS Quantity
        FROM DATA1_OWN.DATA1_STRATEGY DATA1
        LEFT JOIN DATA2_OWN.DATA2_SECURITIES DATA2
          ON DATA2.ID = DATA1.ID
        left join DATA3_own.DATA3_strategy DATA3
            on DATA3.strategy_id = DATA1.STRATEGY_ID
        left join pm_own.pma_sec_attributes_hv DATA4
            on DATA4.ssm_id = DATA1.PARENT_ID
        WHERE DATA1.ACCT_NO IN ({acct})
          AND DATA1.asof_date = TRUNC(DATE '{date}')
          AND DATA1.LT_ID IN (0, 2, 23, 24, 25) AND DATA1.lt_id_2 NOT IN (3, 4, 5)
          AND DATA4.asof_date = TRUNC(DATE '{date}')
          and DATA3.strategy_path_1 = 'NY'
          and DATA3.strategy_path_with_scheme is null
        ORDER BY DATA2.currency_issue
                '''
        df = SQL.query_as_dataframe(sql, conn='PRODUCTION')
        df['STRATEGY'].fillna(value='UNTAGGED', inplace=True)

        return df

    def get_untagged_table_Data(self, date, account, filterPlusOne):

        df = self.get_SQL_untagged(date, account)

        lastEOD_AE = datetime.strptime(date, '%Y-%m-%d')
        lastEOD_AE = lastEOD_AE.strftime('%Y/%m/%d')

        securities = df['SSM_ID'].to_list()

        flt1 = DATA_TOOLReportFilter('strategy_path_1', 'In', ['NY'])  # null

        df_holdings_delta = svc.DATA_TOOL(["OPTION_DELTA*", "UNDL_PRICE", 'PRICE'], securities=securities,
                                   level=DATA_TOOLLevel.Account, filters=[flt1], startDate=lastEOD_AE)
        df_holdings_delta['OPTION_DELTA*'] = pd.to_numeric(df_holdings_delta['OPTION_DELTA*'], errors='coerce')
        df_holdings_delta['UNDL_PRICE'] = pd.to_numeric(df_holdings_delta['UNDL_PRICE'], errors='coerce')
        df_holdings_delta['PRICE'] = pd.to_numeric(df_holdings_delta['PRICE'], errors='coerce')
        df_holdings_delta = df_holdings_delta.rename(columns={'SECURITY_ID': 'SSM_ID'})

        df_holdings_delta = df_holdings_delta.fillna(0)
        df_holdings_delta = df_holdings_delta.groupby(by='SSM_ID').mean().reset_index()

        joined_df = pd.merge(df, df_holdings_delta, on='SSM_ID', how='left')

        joined_df['DELTA'] = joined_df['OPTION_DELTA*'] * joined_df['UNDL_PRICE'] * joined_df['MULTIPLIER'] * joined_df[
            'NO_OF_CONTRACTS']

        joined_df = joined_df[
            ['CURRENCY_ISSUE', 'STRATEGY', 'COMP_SEC_TYPE_CODE', 'MATURITY_DATE', 'PIMCO_DESC', 'DELIVERY_DATE',
             'SSM_ID', 'PCE', 'CURRENCY_EXPOSURE', 'QUANTITY', 'DELTA', 'OPTION_DELTA*']]

        return joined_df

    def mergeAndCompute_untagged(self, filterPlusOne, filterUSD, threshold):
        for account in self.accounts:
            lastEOD = self.getEODDates()[0]
            scndLastEOD = self.getEODDates()[1]

            COMP_SEC_TYPE_CODE = ['OPTN', 'CURR', '']

            pt1 = self.get_untagged_table_Data(lastEOD, account, filterPlusOne)
            pt1 = pt1.groupby(
                by=['CURRENCY_ISSUE', 'STRATEGY', 'COMP_SEC_TYPE_CODE', 'MATURITY_DATE', 'PIMCO_DESC', 'DELIVERY_DATE',
                    'SSM_ID']).agg({'PCE': 'sum', 'CURRENCY_EXPOSURE': 'sum', 'DELTA': 'mean', 'OPTION_DELTA*': 'mean',
                                    'QUANTITY': 'sum'}).reset_index()
            pt1 = pt1[pt1['COMP_SEC_TYPE_CODE'].isin(COMP_SEC_TYPE_CODE)].reset_index(drop=True)
            if filterUSD == True:
                pt1 = pt1[pt1['CURRENCY_ISSUE'] != 'USD']

            pt2 = self.get_untagged_table_Data(scndLastEOD, account, filterPlusOne)
            pt2 = pt2.groupby(
                by=['CURRENCY_ISSUE', 'STRATEGY', 'COMP_SEC_TYPE_CODE', 'MATURITY_DATE', 'PIMCO_DESC', 'DELIVERY_DATE',
                    'SSM_ID']).agg({'PCE': 'sum', 'CURRENCY_EXPOSURE': 'sum', 'DELTA': 'mean', 'OPTION_DELTA*': 'mean',
                                    'QUANTITY': 'sum'}).reset_index()
            pt2 = pt2[pt2['COMP_SEC_TYPE_CODE'].isin(COMP_SEC_TYPE_CODE)].reset_index(drop=True)
            if filterUSD == True:
                pt2 = pt2[pt2['CURRENCY_ISSUE'] != 'USD']

            pt3 = pd.merge(pt1, pt2,
                           on=['CURRENCY_ISSUE', 'STRATEGY', 'COMP_SEC_TYPE_CODE', 'MATURITY_DATE', 'PIMCO_DESC',
                               'DELIVERY_DATE', 'SSM_ID'], how='outer', suffixes=('_last_EOD', '_day_before'))

            pt3[['PCE_last_EOD', 'CURRENCY_EXPOSURE_last_EOD', 'DELTA_last_EOD', 'OPTION_DELTA*_last_EOD',
                 'QUANTITY_last_EOD']] = pt3[
                ['PCE_last_EOD', 'CURRENCY_EXPOSURE_last_EOD', 'DELTA_last_EOD', 'OPTION_DELTA*_last_EOD',
                 'QUANTITY_last_EOD']].fillna(0)

            subtotals = pt3.groupby('CURRENCY_ISSUE').sum().reset_index()
            subtotals = pt3.groupby('CURRENCY_ISSUE').sum().reset_index()
            subtotals['CURRENCY_ISSUE'] = subtotals['CURRENCY_ISSUE']
            subtotals['STRATEGY'] = 'zzzz'
            subtotals['COMP_SEC_TYPE_CODE'] = 'SUBTOTAL'
            subtotals['MATURITY_DATE'] = 'SUBTOTAL'
            subtotals['PIMCO_DESC'] = 'SUBTOTAL'
            subtotals['DELIVERY_DATE'] = 'SUBTOTAL'
            subtotals['SSM_ID'] = 'SUBTOTAL'

            pt3 = pd.concat([pt3, subtotals]).sort_values('CURRENCY_ISSUE', kind='mergesort').reset_index(drop=True)

            global values_columns
            values_columns = pt1.iloc[:, 7:].columns.tolist()

            for i, column in enumerate(values_columns):
                column_lastEOD = pt3.iloc[:, 7 + i]
                column_dayBefore = pt3.iloc[:, 7 + i + len(values_columns)]
                pt3[column + '_diff'] = column_lastEOD - column_dayBefore

            columns = pt3.columns.tolist()
            first_columns = columns[:7]
            last_columns = columns[-len(values_columns):]

            new_columns = first_columns + last_columns + columns[7:-len(values_columns)]
            pt3 = pt3[new_columns]

            global categories
            categories = pt1['CURRENCY_ISSUE'].unique()

            for val in values_columns:
                pt3[f'{val}_diff'].fillna(pt3[f'{val}_last_EOD'], inplace=True)

            pt3 = pt3.groupby(['CURRENCY_ISSUE', 'STRATEGY', 'COMP_SEC_TYPE_CODE']).apply(
                lambda x: x.sort_values('PCE_diff', ascending=False)).reset_index(drop=True)

            pt3.loc[pt3['STRATEGY'] == 'zzzz', 'STRATEGY'] = ''
            pt3.loc[pt3['COMP_SEC_TYPE_CODE'] == 'SUBTOTAL', 'COMP_SEC_TYPE_CODE'] = ''
            pt3.loc[pt3['MATURITY_DATE'] == 'SUBTOTAL', 'MATURITY_DATE'] = ''
            pt3.loc[pt3['PIMCO_DESC'] == 'SUBTOTAL', 'PIMCO_DESC'] = ''
            pt3.loc[pt3['DELIVERY_DATE'] == 'SUBTOTAL', 'DELIVERY_DATE'] = ''

            subtotals = pt3[pt3['SSM_ID'] == 'SUBTOTAL']

            if filterPlusOne == True:
                untagged_currency_filter = subtotals[abs(subtotals['PCE_diff']) > threshold]
                untagged_currency_filter = untagged_currency_filter['CURRENCY_ISSUE'].to_list()
                pt3 = pt3[pt3['CURRENCY_ISSUE'].isin(untagged_currency_filter)].reset_index(drop=True)

            curr_flt = untagged_currency_filter
            if 'USD' in curr_flt:
                curr_flt.remove('USD')
            if len(curr_flt) > 0:
                if len(pt3) > 0:
                    return self.Prettyfy(pt3)

    def run_graphs(self, drill_down, filterUSD, threshold):
        self.get_diff_daily(threshold)
        curr_flt = currency_filter
        if 'USD' in curr_flt:
            curr_flt.remove('USD')
        if len(curr_flt) > 0:
            plots = []
            # plots.append(self.plot(self.get_pce_ot('yes', filterUSD),'PCE by Currency, over time where PCE DIFF > threshold', 'PCE')) #all charts PCE OT
            # plots.append(self.diff_plot(self.get_pce_diff_ot('yes', filterUSD), 'PCE daily difference by Currency, over time where PCE DIFF > threshold','PCE daily diff')) #all charts PCE DIFF OT
            plots.append(self.comparison_plot(self.get_pce_ot('yes', filterUSD), self.get_pce_diff_ot('yes', filterUSD),
                                              'PCE vs PCE daily difference by Currency where PCE DIFF > threshold',
                                              'PCE vs PCE daily diff'))  # all charts PCE DIFF VS PCE OT
            currency = drill_down  # CHANGE FOR DIFFERENT DRILL DOWNS
            if len(currency) > 0:
                # plots.append(self.drill_down(self.get_pce_ot('no', filterUSD),'PCE, over time', 'PCE',currency)) #choose few PCE OT
                # plots.append(self.diff_drill_down(self.get_pce_diff_ot('no', filterUSD), 'PCE daily difference, over time','PCE daily diff',currency)) #choose few PCE DIFF OT
                plots.append(
                    self.comparison_drill_down(self.get_pce_ot('no', filterUSD), self.get_pce_diff_ot('no', filterUSD),
                                               'PCE vs PCE daily difference, over time', 'PCE vs PCE daily diff',
                                               currency))  # choose few PCE DIFF VS PCE OT
            for p in plots:
                return p

    def comparison_plot(self, df_pivot, df_diff, title, pce_label):
        df_merged = pd.merge(df_pivot, df_diff, left_index=True, right_index=True, how='outer')

        nc = len(df_pivot.columns)

        min_pce = df_merged.min().min() - 2
        max_pce = df_merged.max().max() + 2

        num_columns = min(len(df_pivot.columns), 2)
        num_rows = (len(df_pivot.columns) + num_columns - 1) // num_columns

        if nc < 2:
            for i, currency in enumerate(df_pivot.columns):
                fig, ax = plt.subplots(figsize=(9, 4))  # Create a figure and axes
                colors = ['red' if price < 0 else 'blue' for price in df_merged[f'{currency}_y']]
                line = ax.plot(df_pivot.index, df_merged[f'{currency}_x'],
                               color=plt.cm.rainbow(i / len(df_pivot.columns)))
                ax.bar(df_pivot.index, df_merged[f'{currency}_y'], color=colors)
                ax.set_xlabel('Time')
                ax.set_ylabel(pce_label)
                ax.set_title(f'{currency} {pce_label}, Over Time')
                ax.grid(True, linewidth=0.2, axis='y')
                ax.set_ylim(min_pce, max_pce)
                ax.set_xticks([df_pivot.index[0], df_pivot.index[-1]])

                red_patch = mpatches.Patch(color='red', label='-PCE diff')
                blue_patch = mpatches.Patch(color='blue', label='+PCE diff')
                line_legend = mlines.Line2D([], [], color=line[0].get_color(), label='PCE')
                ax.legend(handles=[red_patch, blue_patch, line_legend])

                num_steps = 15
                y_ticks = plt.MaxNLocator(num_steps)
                ax.yaxis.set_major_locator(y_ticks)

                ax.axhline(0, color='black', linewidth=0.5)

                plt.show()
                print('\n\n\n')

        elif nc < 3:
            fig, axes = plt.subplots(nrows=1, ncols=nc, figsize=(18, 4 * num_rows))
            fig.suptitle(title, fontsize=16)
            for i, currency in enumerate(df_pivot.columns):
                colors = ['red' if price < 0 else 'blue' for price in df_merged[f'{currency}_y']]
                line = axes[i].plot(df_pivot.index, df_merged[f'{currency}_x'],
                                    color=plt.cm.rainbow(i / len(df_pivot.columns)))
                axes[i].bar(df_pivot.index, df_merged[f'{currency}_y'], color=colors)
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel(pce_label)
                axes[i].set_title(f'{currency} {pce_label}, Over Time')
                axes[i].set_ylim(min_pce, max_pce)
                axes[i].set_xticks([df_pivot.index[0], df_pivot.index[-1]])
                axes[i].grid(True, linewidth=0.2, axis='y')

                red_patch = mpatches.Patch(color='red', label='-PCE diff')
                blue_patch = mpatches.Patch(color='blue', label='+PCE diff')
                line_legend = mlines.Line2D([], [], color=line[0].get_color(), label='PCE')
                axes[i].legend(handles=[red_patch, blue_patch, line_legend])

                num_steps = 15
                y_ticks = plt.MaxNLocator(num_steps)
                axes[i].yaxis.set_major_locator(y_ticks)

            plt.tight_layout()
            plt.show()
            print('\n\n\n')

        else:

            fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(18, 4 * num_rows))
            fig.suptitle(title, fontsize=16)

            for i, currency in enumerate(df_pivot.columns):
                row = i // num_columns
                col = i % num_columns
                colors = ['red' if price < 0 else 'blue' for price in df_merged[f'{currency}_y']]
                line = axes[row, col].plot(df_pivot.index, df_merged[f'{currency}_x'],
                                           color=plt.cm.rainbow(i / len(df_pivot.columns)))
                axes[row, col].bar(df_pivot.index, df_merged[f'{currency}_y'], color=colors)
                axes[row, col].set_xlabel('Time')
                axes[row, col].set_ylabel(pce_label)
                axes[row, col].set_title(f'{currency} {pce_label}, Over Time')
                axes[row, col].set_ylim(min_pce, max_pce)
                axes[row, col].set_xticks([df_pivot.index[0], df_pivot.index[-1]])
                axes[row, col].grid(True, linewidth=0.2, axis='y')

                num_steps = 15
                y_ticks = plt.MaxNLocator(num_steps)
                axes[row, col].yaxis.set_major_locator(y_ticks)

                red_patch = mpatches.Patch(color='red', label='-PCE diff')
                blue_patch = mpatches.Patch(color='blue', label='+PCE diff')
                line_legend = mlines.Line2D([], [], color=line[0].get_color(), label='PCE')
                axes[row, col].legend(handles=[red_patch, blue_patch, line_legend])

            if len(df_pivot.columns) < num_rows * num_columns:
                for i in range(len(df_pivot.columns), num_rows * num_columns):
                    fig.delaxes(axes.flatten()[i])

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()
            print('\n\n\n')

        return fig

    def maturities_plot(self):
        pt3_bar = self.mergeAndCompute(filterPlusFive=True, filterUSD=True, threshold=5, barchart='yes')

        currencies = pt3_bar['CURRENCY_ISSUE'].unique().tolist()

        if len(currencies) > 0:
            n = len(currencies)
            ncols = 2
            nrows = n // ncols + n % ncols
            pos = range(1, n + 1)

            fig = plt.figure(figsize=(18, 3 * n))
            fig.suptitle('Top 10 absolute PCE daily difference by Maturity, for Currencies where PCE Diff > threshold',
                         fontsize=16)

            for i, curr in enumerate(currencies):

                pt3_bar_chart = pt3_bar[pt3_bar['CURRENCY_ISSUE'] == curr]
                df = pt3_bar_chart.groupby(['MATURITY_DATE']).sum().reset_index()
                df = df.dropna()

                if len(df) > 10:
                    df = df.loc[df['PCE_diff'].abs().nlargest(10).index]

                else:
                    df = df.sort_values('PCE_diff', ascending=False)

                df = df[['MATURITY_DATE', 'PCE_diff', 'PCE_last_EOD', 'PCE_day_before']]
                df['MATURITY_DATE'] = pd.to_datetime(df['MATURITY_DATE']).dt.strftime('%Y-%m-%d')

                categories = df['MATURITY_DATE'].tolist()
                group_names = ['PCE_diff', 'PCE_last_EOD', 'PCE_day_before']
                values = df[group_names].values.T
                bar_width = 0.25
                bar_positions = np.arange(len(categories))
                ax = fig.add_subplot(nrows, ncols, pos[i])
                colors = ['#8cfa75', '#75a4fa', '#9dbefa']
                for i, group_values in enumerate(values):
                    bar_positions_adjusted = bar_positions + (i * bar_width)
                    ax.bar(bar_positions_adjusted, group_values, bar_width, label=group_names[i], color=colors[i])
                ax.set_xticks(bar_positions)
                ax.set_xticklabels(categories, rotation=45, ha='right')
                ax.set_xlabel('Maturities')
                ax.set_ylabel('PCE')
                ax.set_title('Top 10 absolute PCE daily difference by Maturity for ' + curr)
                ax.legend()
                ax.yaxis.grid(True, linestyle='--', alpha=0.6)
                for i in range(len(categories)):
                    ax.axvline(i - 0.25, color='#a3a3a3', linestyle='--', alpha=0.2)

            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.show()

            return fig

        else:
            return None

    def get_report(self):
        small_table = self.run_job1(threshold=5)
        time_series = self.run_graphs(drill_down=[], filterUSD=True, threshold=5)
        maturities = self.maturities_plot()
        untagged = self.mergeAndCompute_untagged(filterPlusOne=True, filterUSD=True, threshold=1)
        PCE_breakdown = self.mergeAndCompute(filterPlusFive=True, filterUSD=True, threshold=5)

        def number_to_letter(i):
            return chr(i + 65)

        pr = REPORTS().title("Daily difference in Currency PCE, and breakdown of change") \
            .inc(small_table) \
            .linebreak()

        if time_series is None:
            pr.para('No PCE changes above threshold today') \
                .linebreak()
        else:
            pr.fig(time_series)

        if maturities is None:
            pr.para('No PCE changes above threshold today') \
                .linebreak()
        else:
            pr.fig(maturities)

        if untagged is None:
            pr.title("Breakdown of PCE change on UNTAGGED strategy securities, above threshold") \
                .para('No Untagged above threshhold today') \
                .linebreak()
        else:
            pr.spacer('15px') \
                .title("Breakdown of PCE change on UNTAGGED strategy securities, above threshold") \
                .linebreak() \
                .inc(untagged)

        if PCE_breakdown is None:
            pr.title("Breakdown of PCE change, grouped, and above threshold") \
                .para('No Currencies above threshhold today') \
                .linebreak()
        else:
            pr.spacer('25px') \
                .title("Breakdown of PCE change, grouped, and above threshold") \
                .linebreak() \
                .inc(PCE_breakdown)

            file_path = f'/home/{getpass.getuser()}/PCE_breakdown.xlsx'

            column_names = CurrAggregator().get_columns()

            global xlreport
            xlreport = REPORTSXl()

            xlreport.sheet_name('PCE_breakdown').inc(PCE_breakdown)

            for i, name in enumerate(column_names):
                length = len(name)
                letter = number_to_letter(i)
                xlreport.gcws().column_dimensions[f'{letter}'].width = max(length * 1.5, (1 / (length)) * 100)

            try:
                xlreport.add_sheet('untagged').inc(untagged)
                for i, name in enumerate(column_names):
                    length = len(name)
                    letter = number_to_letter(i)
                    xlreport.gcws().column_dimensions[f'{letter}'].width = max(length * 1.5, (1 / (length)) * 100)
            except Exception:
                pass

            xlreport.save(file_path)

        global html_buffer
        html_buffer = StringIO()
        html = pr.to_html()
        html_buffer.write(html)

        return pr

    def report_attachments(self):
        return xlreport, html_buffer
from PIMCO.PROPERTY.SQL import SQL
from PIMCO.PROPERTY.REST import FORMAT, REST
import json
from flask import Flask, make_response
import PIMCO.PROPERTY.FLASK
from PIMCO.PROPERTY.FLASK import RESOURCE, PARSER
from PIMCO.PROPERTY.TABLES.UTILITIES import DASH_SERVICE
import dash
from PIMCO.PROPERTY.DATA_TOOL import *
import pandas as pd
import numpy as np

server = Flask(__name__)
api = PIMCO.PROPERTY.FLASK.Api(server)
DASH_SERVICE(api, base_url="")

app = dash.Dash(server=server)
app.config.suppress_callback_exceptions = True


@api.RESOURCE('/test')
class ptxl_test(RESOURCE):
    def get(self):
        svc = DATA_TOOL('prod')
        p = PARSER.RequestParser()
        p.add_argument('Special')
        p.add_argument('PM1')
        p.add_argument('Accounts')
        p.add_argument('Fields')
        p.add_argument('CUSIP')
        p.add_argument('TargetCat')
        p.add_argument('Amount')
        p.add_argument('ChangeOrSet')
        p.add_argument('Price')
        p.add_argument('Broker')
        p.add_argument('Block Amount')
        p.add_argument('Bring down to Target')
        p.add_argument('Hard UPI Limit')
        p.add_argument('BuySell')
        p.add_argument('Risks')
        p.add_argument('Currency')
        p.add_argument('MaxDWE')
        p.add_argument('MaxUPI')
        p.add_argument('Amt_bought')
        p.add_argument('EXCEL_TEST')
        a = p.parse_args()

        print(a)

        Special = a.get('Special')
        Special = list(map(str.strip, Special.split(','))) if Special else []

        PM1 = a.get('PM1')
        PM1 = list(map(str.strip, PM1.split(','))) if PM1 else []

        Accounts = a.get('Accounts')
        Accounts = list(map(str.strip, Accounts.split(','))) if Accounts else []

        Fields = a.get('Fields')
        Fields = list(map(str.strip, Fields.split(','))) if Fields else []

        CUSIP = a.get('CUSIP')
        CUSIP = list(map(str.strip, CUSIP.split(','))) if CUSIP else []

        TargetCat = a.get('TargetCat')
        TargetCat = list(map(str.strip, TargetCat.split(','))) if TargetCat else []

        Amount = a.get('Amount')
        Amount = list(map(str.strip, Amount.split(','))) if Amount else []

        ChangeOrSet = a.get('ChangeOrSet')
        ChangeOrSet = list(map(str.strip, ChangeOrSet.split(','))) if ChangeOrSet else []

        Price = a.get('Price')
        Price = list(map(str.strip, Price.split(','))) if Price else []

        Broker = a.get('Broker')
        Broker = list(map(str.strip, Broker.split(','))) if Broker else []

        Block = a.get('Block Amount')
        Block = list(map(str.strip, Block.split(','))) if Block else []

        BDTT = a.get('Bring down to Target')
        BDTT = list(map(str.strip, BDTT.split(','))) if BDTT else []

        UPILimit = a.get('Hard UPI Limit')
        UPILimit = list(map(str.strip, UPILimit.split(','))) if UPILimit else []

        BuySell = a.get('BuySell')
        BuySell = list(map(str.strip, BuySell.split(','))) if BuySell else []

        Risks = a.get('Risks')
        Risks = list(map(str.strip, Risks.split(','))) if Risks else []

        Currency = a.get('Currency')
        Currency = list(map(str.strip, Currency.split(','))) if Currency else []

        MaxDWE = a.get('MaxDWE')
        MaxDWE = list(map(str.strip, MaxDWE.split(','))) if MaxDWE else []

        MaxUPI = a.get('MaxUPI')
        MaxUPI = list(map(str.strip, MaxUPI.split(','))) if MaxUPI else []

        Amt_bought = a.get('Amt_bought')
        Amt_bought = list(map(str.strip, Amt_bought.split(','))) if Amt_bought else []

        EXCEL_TEST = a.get('EXCEL_TEST')

        def pm_names():
            sql = '''
                select UNIQUE(PM1)
                from ACCTS
                '''
            df = SQL.query_as_dataframe(sql, conn='PRODUCTION')
            df = df.sort_values(by='PM1')
            return df

        def if_pm(
            PM1, Fields, CUSIP, TargetCat, Amount, ChangeOrSet, Price,
            Broker, Block, BDTT, UPILimit, BuySell, Risks,
            Currency, MaxDWE, MaxUPI, EXCEL_TEST
        ):
            df_long = pd.DataFrame()
            pm_str = ', '.join(f"'{pm}'" for pm in PM1)

            # Use the IN clause in SQL to get information for all accounts in the list
            sql = f'''
                SELECT PM1, ACCT_NO
                FROM ACCTS
                WHERE PM1 IN ({pm_str})
            '''

            df = SQL.query_as_dataframe(sql, conn='PRODUCTION')

            accts = df['ACCT_NO'].tolist()

            dynamic_categories = svc.DATA_TOOL(
                Fields, accounts=accts, level=DATA_TOOL.Var
            )

            df = pd.merge(df, dynamic_categories, on='ACCT_NO')

            fx_sql = f"""
            SELECT currency_issue, exchange_rate
            FROM COUNTRY
            WHERE CURRENCY_ISSUE = '{Currency[0]}'
            """
            query_result = SQL.query_as_dataframe(sql, conn='PRODUCTION')
            fx_sql = 1 / float(query_result['EXCHANGE_RATE'])

            df['MKT_VAL'] = df['MKT_VAL'] / 10**6 / fx_sql

            flt_pmv_by_upi = DATA_TOOL('ult_parent_id', 'In', CUSIP, by_security=True)
            pmv_by_upi_df = svc.DATA_TOOL(
                accounts=accts, securities=CUSIP,
                fields=['pmv_by_upi'], level=DATA_TOOL.Var, filters=[flt_pmv_by_upi]
            )[['ACCT_NO', 'SECURITY', 'PMV_BY_UPI']]

            df = pd.merge(df, pmv_by_upi_df, on='ACCT_NO')

            if TargetCat[0] == 'UPI':
                flt_pmv_by_upi = DATA_TOOL('ult_parent_id', 'In', CUSIP, by_security=True)
                pmv_by_upi_df = svc.DATA_TOOL(
                    accounts=accts, securities=CUSIP,
                    fields=['pmv_by_upi'], level=DATA_TOOL.Var, filters=[flt_pmv_by_upi]
                )[['ACCT_NO', 'PMV_BY_UPI']]
                pmv_by_upi_df = pmv_by_upi_df.rename(columns={'PMV_BY_UPI': 'TargetCat'})

                df = pd.merge(df, pmv_by_upi_df, on='ACCT_NO')
            else:
                values_dict = {'DWE': 'DWE', 'MWS_NEW': 'MWS_NEW'}
                lookup_value = values_dict.get(TargetCat[0])
                TargetCat_df = svc.DATA_TOOL(
                    accounts=accts,
                    fields=lookup_value, level=DATA_TOOL.Var
                )[['ACCT_NO', TargetCat[0]]]
                TargetCat_df = TargetCat_df.rename(columns={TargetCat[0]: 'TargetCat'})

                df = pd.merge(df, TargetCat_df, on='ACCT_NO')

            svc_pc = DATA_TOOL('prod')

            def spnd_amt(row):
                res = svc_pc.DATA_TOOL(row['ACCT_NO'], row['SECURITY'], [1], ["BUY"])
                rules = res.rules
                if len(rules) > 0:  # test at the end
                    spnd = [item['spendAmt'] for item in rules if 'spendAmt' in item][0]
                    return spnd / (10 ** 6) / fx_sql
                else:
                    return row['MKT_VAL'] / fx_sql

            df['ACE_SPEND'] = df.apply(spnd_amt, axis=1)

            df['Spend_PMV'] = df['ACE_SPEND'] / df['MKT_VAL'] * 100

            def PMV_Space(row, MaxUPI):
                return min(max(float(MaxUPI[0]) - row['PMV_BY_UPI'], 0), row['Spend_PMV'])

            df['PMV_Space'] = df.apply(PMV_Space, args=(MaxUPI,), axis=1)

            def Impact(row, ChangeOrSet_amt, set_chng_amt):
                if ChangeOrSet_amt == "CHANGE":
                    return set_chng_amt
                elif ChangeOrSet_amt == "SET":
                    return set_chng_amt - row['TargetCat']
                else:
                    return 0

            ChangeOrSet_amt = ChangeOrSet[0]
            set_chng_amt = float(Amount[0])
            df['Impact'] = df.apply(Impact, args=(ChangeOrSet_amt, set_chng_amt), axis=1)

            def PMV_TO_ADD(row, TargetCat_pta, fx_sql):
                if TargetCat_pta == "UPI":
                    value = (row['Impact'] / 100)
                    value = value * (
                        100 / float(svc.DATA_TOOL(["PRICE"], securities=row['SECURITY'])["PRICE"][0])
                    )
                else:
                    values_dict = {'DWE': 'DURATION', 'MWS_NEW': 'MWS_NEW*'}
                    lookup_value = values_dict.get(TargetCat_pta)
                    adjusted_price_key = ["ADJUSTED_PRICE"]
                    adjusted_price_security = row['SECURITY']
                    adjusted_price_result = svc.DATA_TOOL(
                        adjusted_price_key, securities=adjusted_price_security
                    )
                    adjusted_price_value = adjusted_price_result["ADJUSTED_PRICE"][0]

                    # Similarly for other variables
                    lookup_value_key = [lookup_value]
                    lookup_value_security = row['SECURITY']
                    lookup_value_result = svc.DATA_TOOL(
                        lookup_value_key, securities=lookup_value_security
                    )
                    lookup_value_value = lookup_value_result[lookup_value][0]

                    exchange_rate_key = ["EXCHANGE_RATE"]
                    exchange_rate_security = row['SECURITY']
                    exchange_rate_result = svc.DATA_TOOL(
                        exchange_rate_key, securities=exchange_rate_security
                    )
                    exchange_rate_value = exchange_rate_result["EXCHANGE_RATE"][0]

                    value = row['Impact'] / (
                        float(lookup_value_value) * float(adjusted_price_value) / 100 / (
                            float(exchange_rate_value) / fx_sql
                        )
                    )
                return max(min(row['PMV_Space'], value * 100), 0)

            # Max_DWE_pta = float(Max_DWE[0])
            TargetCat_pta = TargetCat[0]
            # Max_DWE_pta = VALUES_LIST
            df['PMV_TO_ADD'] = df.apply(PMV_TO_ADD, axis=1, args=(TargetCat_pta, fx_sql))

            df['Notional'] = df.apply(
                lambda row: (row['PMV_TO_ADD'] / 100) * (row['MKT_VAL'] * 10 ** 6), axis=1
            )

            df['Notional'] = df['Notional'].div(100000).round().mul(100000)

            df = df.replace(0, np.nan)

            df_long = pd.concat([df_long, df], ignore_index=True)

            df_long['Allo_Amt'] = df_long.apply(
                lambda row: row['Notional'] if len(Amt_bought) == 0
                else (
                    float(Amt_bought[0]) * row['Notional']
                ) / df_long['Notional'].sum(),
                axis=1
            )

            df_long['Allo_Amt'] = df_long['Allo_Amt'].div(100000).round().mul(100000)

            def Ace(row):
                return f'=ace_status("{row["ACCT_NO"]}",\
"{row["SECURITY"]}","{row["Allo_Amt"]}",,,"{Price[0]}",,,)'

            df_long['Ace Status'] = df_long.apply(Ace, axis=1)

            def Trade_Impact_DWE(row, TargetCat):
                return f'=IF("{TargetCat[0]}"="UPI",{row["PMV_TO_ADD"]}*(DATA_TOOL\
({row["SECURITY"]},"price")/100),DATA_TOOL({row["ACCT_NO"]},{row["SECURITY"]},\
{row["Allo_Amt"]},{TargetCat[0]},0))'

            df_long['Trade_Impact_DWE'] = df_long.apply(Trade_Impact_DWE, args=(TargetCat), axis=1)

            return df_long

        def if_acct(
            PM1, Fields, CUSIP, TargetCat, Amount, ChangeOrSet, Price,
            Broker, Block, BDTT, UPILimit, BuySell, Risks,
            Currency, MaxDWE, MaxUPI, EXCEL_TEST
        ):
            df_long = pd.DataFrame()

            accounts_str = ', '.join(f"'{acct}'" for acct in Accounts)

            # Use the IN clause in SQL to get information for all accounts in the list
            sql = f'''
                SELECT PM1, ACCT_NO
                FROM ACCTS
                WHERE ACCT_NO IN ({accounts_str})
            '''

            df = SQL.query_as_dataframe(sql, conn='PRODUCTION')

            accts = df['ACCT_NO'].tolist()

            dynamic_categories = svc.DATA_TOOL(
                Fields, accounts=accts, level=DATA_TOOL.Var
            )

            df = pd.merge(df, dynamic_categories, on='ACCT_NO')

            fx_sql = f"""
            SELECT currency_issue, exchange_rate
            FROM COUNTRY
            WHERE CURRENCY_ISSUE = '{Currency[0]}'
            """
            query_result = SQL.query_as_dataframe(sql, conn='PRODUCTION')
            fx_sql = 1 / float(query_result['EXCHANGE_RATE'])

            df['MKT_VAL'] = df['MKT_VAL'] / 10**6 / fx_sql

            flt_pmv_by_upi = DATA_TOOL('ult_parent_id', 'In', CUSIP, by_security=True)
            pmv_by_upi_df = svc.DATA_TOOL(
                accounts=accts, securities=CUSIP,
                fields=['pmv_by_upi'], level=DATA_TOOL.Var, filters=[flt_pmv_by_upi]
            )[['ACCT_NO', 'SECURITY', 'PMV_BY_UPI']]

            df = pd.merge(df, pmv_by_upi_df, on='ACCT_NO')

            if TargetCat[0] == 'UPI':
                flt_pmv_by_upi = DATA_TOOL('ult_parent_id', 'In', CUSIP, by_security=True)
                pmv_by_upi_df = svc.DATA_TOOL(
                    accounts=accts, securities=CUSIP,
                    fields=['pmv_by_upi'], level=DATA_TOOL.Var, filters=[flt_pmv_by_upi]
                )[['ACCT_NO', 'PMV_BY_UPI']]
                pmv_by_upi_df = pmv_by_upi_df.rename(columns={'PMV_BY_UPI': 'TargetCat'})

                df = pd.merge(df, pmv_by_upi_df, on='ACCT_NO')
            else:
                values_dict = {'DWE': 'DWE', 'MWS_NEW': 'MWS_NEW'}
                lookup_value = values_dict.get(TargetCat[0])
                TargetCat_df = svc.DATA_TOOL(
                    accounts=accts,
                    fields=lookup_value, level=DATA_TOOL.Var
                )[['ACCT_NO', TargetCat[0]]]
                TargetCat_df = TargetCat_df.rename(columns={TargetCat[0]: 'TargetCat'})

                df = pd.merge(df, TargetCat_df, on='ACCT_NO')

            svc_pc = DATA_TOOL('prod')

            def spnd_amt(row):
                res = svc_pc.ace_status(row['ACCT_NO'], row['SECURITY'], [1], ["BUY"])
                rules = res.rules
                if len(rules) > 0:  # test at the end
                    spnd = [item['spendAmt'] for item in rules if 'spendAmt' in item][0]
                    return spnd / (10 ** 6) / fx_sql
                else:
                    return row['MKT_VAL'] / fx_sql

            df['ACE_SPEND'] = df.apply(spnd_amt, axis=1)

            df['Spend_PMV'] = df['ACE_SPEND'] / df['MKT_VAL'] * 100

            def PMV_Space(row, MaxUPI):
                return min(max(float(MaxUPI[0]) - row['PMV_BY_UPI'], 0), row['Spend_PMV'])

            df['PMV_Space'] = df.apply(PMV_Space, args=(MaxUPI,), axis=1)

            def Impact(row, ChangeOrSet_amt, set_chng_amt):
                if ChangeOrSet_amt == "CHANGE":
                    return set_chng_amt
                elif ChangeOrSet_amt == "SET":
                    return set_chng_amt - row['TargetCat']
                else:
                    return 0

            ChangeOrSet_amt = ChangeOrSet[0]
            set_chng_amt = float(Amount[0])
            df['Impact'] = df.apply(Impact, args=(ChangeOrSet_amt, set_chng_amt), axis=1)

            def PMV_TO_ADD(row, TargetCat_pta, fx_sql):
                if TargetCat_pta == "UPI":
                    value = (row['Impact'] / 100)
                    value = value * (
                        100 / float(svc.DATA_TOOL(["PRICE"], securities=row['SECURITY'])["PRICE"][0])
                    )
                else:
                    values_dict = {'DWE': 'DURATION', 'MWS_NEW': 'MWS_NEW*'}
                    lookup_value = values_dict.get(TargetCat_pta)
                    adjusted_price_key = ["ADJUSTED_PRICE"]
                    adjusted_price_security = row['SECURITY']
                    adjusted_price_result = svc.DATA_TOOL(
                        adjusted_price_key, securities=adjusted_price_security
                    )
                    adjusted_price_value = adjusted_price_result["ADJUSTED_PRICE"][0]

                    # Similarly for other variables
                    lookup_value_key = [lookup_value]
                    lookup_value_security = row['SECURITY']
                    lookup_value_result = svc.DATA_TOOL(
                        lookup_value_key, securities=lookup_value_security
                    )
                    lookup_value_value = lookup_value_result[lookup_value][0]

                    exchange_rate_key = ["EXCHANGE_RATE"]
                    exchange_rate_security = row['SECURITY']
                    exchange_rate_result = svc.DATA_TOOL(
                        exchange_rate_key, securities=exchange_rate_security
                    )
                    exchange_rate_value = exchange_rate_result["EXCHANGE_RATE"][0]

                    value = row['Impact'] / (
                        float(lookup_value_value) * float(adjusted_price_value) / 100 / (
                            float(exchange_rate_value) / fx_sql
                        )
                    )
                return max(min(row['PMV_Space'], value * 100), 0)

            # Max_DWE_pta = float(Max_DWE[0])
            TargetCat_pta = TargetCat[0]
            # Max_DWE_pta = VALUES_LIST
            df['PMV_TO_ADD'] = df.apply(PMV_TO_ADD, axis=1, args=(TargetCat_pta, fx_sql))

            df['Notional'] = df.apply(
                lambda row: (row['PMV_TO_ADD'] / 100) * (row['MKT_VAL'] * 10 ** 6), axis=1
            )

            df['Notional'] = df['Notional'].div(100000).round().mul(100000)

            df = df.replace(0, np.nan)

            df_long = pd.concat([df_long, df], ignore_index=True)

            df_long['Allo_Amt'] = df_long.apply(
                lambda row: row['Notional'] if len(Amt_bought) == 0
                else (
                    float(Amt_bought[0]) * row['Notional']
                ) / df_long['Notional'].sum(),
                axis=1
            )

            df_long['Allo_Amt'] = df_long['Allo_Amt'].div(100000).round().mul(100000)

            def Ace(row):
                return f'=ace_status("{row["ACCT_NO"]}",\
"{row["SECURITY"]}","{row["Allo_Amt"]}",,,"{Price[0]}",,,)'

            df_long['Ace Status'] = df_long.apply(Ace, axis=1)

            def Trade_Impact_DWE(row, TargetCat):
                return f'=IF("{TargetCat[0]}"="UPI",{row["PMV_TO_ADD"]}*(DATA_TOOL\
({row["SECURITY"]},"price")/100),DATA_TOOL({row["ACCT_NO"]},{row["SECURITY"]},\
{row["Allo_Amt"]},{TargetCat[0]},0))'

            df_long['Trade_Impact_DWE'] = df_long.apply(Trade_Impact_DWE, args=(TargetCat), axis=1)

            return df_long

        df_final = pd.DataFrame()
        if Special[0] == 'pm':
            i = pm_names()
            df_final = pd.concat([df_final, i], ignore_index=True)

        else:
            if len(PM1) > 0:
                i = if_pm(
                    PM1, Fields, CUSIP, TargetCat, Amount, ChangeOrSet, Price,
                    Broker, Block, BDTT, UPILimit, BuySell, Risks,
                    Currency, MaxDWE, MaxUPI, EXCEL_TEST
                )
                df_final = pd.concat([df_final, i], ignore_index=True)

            if len(Accounts) > 0:
                i = if_acct(
                    PM1, Fields, CUSIP, TargetCat, Amount, ChangeOrSet, Price,
                    Broker, Block, BDTT, UPILimit, BuySell, Risks,
                    Currency, MaxDWE, MaxUPI, EXCEL_TEST
                )
                df_final = pd.concat([df_final, i], ignore_index=True)

            df_final['EXCEL_OUTPUT'] = EXCEL_TEST

            df_final = df_final.replace(np.nan, 0)

        resp = make_response(json.dumps(FORMAT.from_df(df_final)))
        resp.headers['Content-Type'] = REST.FORMAT
        return resp


@api.RESOURCE('/test2')
class ptxl_test2(RESOURCE):
    def get(self):
        p = PARSER.RequestParser()

        p.add_argument('PM1')
        p.add_argument('Accounts')
        p.add_argument('securities')
        a = p.parse_args()

        print(a)

        PM1 = a.get('PM1')
        PM1 = list(map(str.strip, PM1.split(','))) if PM1 else []

        Accounts = a.get('Accounts')
        Accounts = list(map(str.strip, Accounts.split(','))) if Accounts else []

        securities = a.get('securities')
        securities = list(map(str.strip, securities.split(','))) if securities else []

        def populate(
            PM1, Accounts, securities
        ):
            df = pd.DataFrame()

            if len(PM1) > 0:
                pm_str = ', '.join(f"'{pm}'" for pm in PM1)
                sql = f'''
                    SELECT PM1, ACCT_NO
                    FROM ACCTS
                    WHERE PM1 IN ({pm_str})
                '''

                df_pm = SQL.query_as_dataframe(sql, conn='PRODUCTION')
                df_pm['PM1'] = pd.Categorical(df_pm['PM1'], categories=PM1, ordered=True)
                df_pm = df_pm.sort_values('PM1')
                df = pd.concat([df, df_pm], ignore_index=True)

            if len(Accounts) > 0:
                accounts_str = ', '.join(f"'{acct}'" for acct in Accounts)
                sql = f'''
                    SELECT PM1, ACCT_NO
                    FROM ACCTS
                    WHERE ACCT_NO IN ({accounts_str})
                    '''

                df_acct = SQL.query_as_dataframe(sql, conn='PRODUCTION')
                df_acct['ACCT_NO'] = df_acct['ACCT_NO'].astype('str')
                df_acct['ACCT_NO'] = pd.Categorical(
                    df_acct['ACCT_NO'], categories=Accounts, ordered=True
                )
                df_acct = df_acct.sort_values('ACCT_NO')
                df = pd.concat([df, df_acct], ignore_index=True)

            df = (
                df.assign(key=1)
                .merge(
                    pd.DataFrame({'SECURITY': securities, 'key': 1}),
                    on='key'
                )
                .drop('key', axis=1)
            )

            df = df[['SECURITY', 'PM1', 'ACCT_NO']]
            return df

        df_final = populate(PM1, Accounts, securities)

        resp = make_response(json.dumps(FORMAT.from_df(df_final)))
        resp.headers['Content-Type'] = REST.FORMAT
        return resp
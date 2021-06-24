###################################################
#Rating Product & Sorting Reviews in Amazon
###################################################

import numpy as np
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)




df_ = pd.read_csv("df_sub.csv")
df_sub = df_.copy()
df_sub.head(10)
df_sub.shape
df_sub.info()
df_sub.isnull().sum()
df_sub.describe().T




df_sub["overall"].mean()



df_sub.info()

df_sub["reviewTime"] = pd.to_datetime(df_sub["reviewTime"], dayfirst=True)

df_sub["reviewTime"].max()
current_time = pd.to_datetime("2014-12-08 0:0:0")
df_sub["day_diff"] = (current_time - df_sub["reviewTime"]).dt.days

a = df_sub["day_diff"].quantile(0.25)
b = df_sub["day_diff"].quantile(0.50)
c = df_sub["day_diff"].quantile(0.75)


df_sub.head()
df_sub.info()
df_sub["day_diff"].dtypes

df_sub.loc[df_sub["day_diff"] <= a, "overall"].mean() * 30 / 100 + \
    df_sub[(df_sub["day_diff"] > a) & df_sub["day_diff"] <= b].mean() * 24 / 100 + \
    df_sub[(df_sub["day_diff"] > b) & df_sub["day_diff"] <= c].mean() * 24 / 100 + \
    df_sub[df_sub["day_diff"] > c].mean() * 22 / 100




df_sub.info()

df_sub["helpful_yes"] = df_sub[["helpful"]].applymap(lambda x: x.split(",")[0].strip('[')).astype(int)

df_sub["helpful_total_vote"] = df_sub[["helpful"]].applymap(lambda x: x.split(",")[1].strip(']')).astype(int)


df_sub["helpful_no"] = df_sub["helpful_total_vote"] - df_sub["helpful_yes"]




def score_pos_neg_diff(pos, neg):
    return pos - neg

df_sub["score_pos_neg_diff"] = df_sub.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                         x["helpful_no"]),
                                            axis=1)
df_sub.sort_values("score_pos_neg_diff",ascending=False)

df_sub[["helpful_total_vote","helpful_yes","helpful_no","score_pos_neg_diff"]].head(35)



def score_average_rating(pos, neg):
    if pos - neg == 0:
        return 0
    else:
        return pos / (pos + neg)


df_sub["score_average_rating"] = df_sub.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                             x["helpful_no"]),
                                            axis=1)



def wilson_lower_bound(pos, neg, confidence=0.95):

    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df_sub["wilson_lower_bound"] = df_sub.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                         x["helpful_no"]),
                                            axis=1)


df_sub.info()

df_sub[["overall","helpful_total_vote","helpful_yes","helpful_no","score_pos_neg_diff","score_average_rating","wilson_lower_bound"]].sort_values("score_pos_neg_diff", ascending=False).head(20)




















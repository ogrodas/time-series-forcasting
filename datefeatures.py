import pandas as pd
import datetime
from datetime import date
from datetime import timedelta

from pandas.api.types import CategoricalDtype
import fastai.tabular


public_holidays=['påskedag',
 'palmesøndag',
 'skjærtorsdag',
 'andre_påskedag',
 'langfredag',
 'kristi_himmelfartsdag',
 'pinsedag',
 'andre_pinsedag',
 'nyttårsdag',
 'arbeiderenesdag',
 'grunnlovsdag',
 'juledag',
 'andrejuledag']

cat_public_holidays = CategoricalDtype(categories=public_holidays,ordered=True)

def calc_easter(year):
    """
    An implementation of Butcher's Algorithm for determining the date of Easter for the Western church. Works for any date in the Gregorian calendar (1583 and onward). 
    Returns a date object.Returns Easter as a date object.
    http://code.activestate.com/recipes/576517-calculate-easter-western-given-a-year/
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = (19 * a + b - b // 4 - ((b - (b + 8) // 25 + 1) // 3) + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - (c % 4)) % 7
    f = d + e - 7 * ((a + 11 * d + 22 * e) // 451) + 114
    month = f // 31
    day = f % 31 + 1    
    return date(year, month, day)


def get_public_holidays_year(year):
    h=dict()

    #Beveglige
    h["påskedag"]=calc_easter(year)    
    h["palmesøndag"]=h["påskedag"]-timedelta(days=7)
    h["skjærtorsdag"]=h["påskedag"]-timedelta(days=3)
    h["andre_påskedag"]=h["påskedag"]+timedelta(days=1)
    h["langfredag"]=h["skjærtorsdag"] + timedelta(days=1)
    h["kristi_himmelfartsdag"]=h["påskedag"]+timedelta(days=39)
    h["pinsedag"]=h["påskedag"]+timedelta(days=49)
    h["andre_pinsedag"]=h["påskedag"]+timedelta(days=50)
    
    #Faste
    h["nyttårsdag"]=date(year,1,1)
    h["arbeiderenesdag"]=date(year,5,1)
    h["grunnlovsdag"]=date(year,5,17)
    h["juledag"]=date(year,12,25)
    h["andrejuledag"]=date(year,12,26)

    df=pd.DataFrame(h.items(),columns=["public_holiday_name","date"])
    df["public_holiday_name"]=df.public_holiday_name.astype(cat_public_holidays)
    df["date"]=pd.to_datetime(df.date)
    return df

def get_public_holidays(years):
    dfs=[]
    for year in years:
        dfs.append(get_public_holidays_year(year))
    return pd.concat(dfs)

def add_public_holidays(df,col):
    years=df[col].dt.year.unique()
    holidays=get_public_holidays(years)
    r=df.merge(holidays,left_on=col,right_on="date",how="left")
    if col!="date":
        r=r.drop("date",axis=1)
    r["public_holiday"]=r.public_holiday_name.notna()
    return r
    

def add_rolling_datefeatures(df,col):
    """Depends on public_holidays function above for df.publi_oliday"""
    df["workday"]=(df[col].dt.dayofweek.isin([0,1,2,3,4]) & ~df.public_holiday)
    df["freeday"]=~df.workday
    df["dummy_group"]=1
    df=fastai.tabular.add_elapsed_times(df,"freeday",col,"dummy_group")
    #data=fastai.tabular.add_elapsed_times(data,"public_holiday","Dato","dummy_group")
    df=df.drop("dummy_group",axis=1)
    df["inneklemt"]=(df.Afterfreeday==1) & (df.Beforefreeday==-1)
    return df


def generate_date_features(start,end):
    df=pd.date_range(start=start, end=end).to_frame(name="date").reset_index(drop=True)
    fastai.tabular.add_datepart(df,"date",drop=False)
    df=add_public_holidays(df,"date")
    df=add_rolling_datefeatures(df,"date")
    return df


if __name__=="__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("startdate",help="Start of time series")
    parser.add_argument("enddate",help="End of time series")
    parser.add_argument("-c", "--csv", action="store_true",help="Output CSV")
    args = parser.parse_args()

    df=generate_date_features(args.startdate,args.enddate)
    if args.csv:
        df.to_csv(sys.stdout)
    else:
        print (df)

#df=generate_date_features("2000-01-01","2029-12-31")
#df.to_csv("date-features.csv")
#df.head()
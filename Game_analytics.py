#!/usr/bin/env python
# coding: utf-8

# In[40]:


#Для начала импортируем возможные библиотеки которые будем использовать при дальнейшей работе.
import pandas as pd
import os
import numpy as np
import datetime
import vk_api
import requests
import json
import random
import scipy.stats as ss

from statsmodels.stats.multicomp import (pairwise_tukeyhsd)
import statsmodels.formula.api as smf
import statsmodels.api as sm

import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.express as px

import sys

from io import BytesIO
from datetime import timedelta
import requests
sns.set(style="whitegrid")


# In[ ]:





# In[2]:


#Считываем файлы 
customer_df = pd.read_csv("/mnt/HC_Volume_18315164/home-jupyter/jupyter-b-kugotov/Папка с данными для проектов/olist_customers_dataset.csv")
orders_df = pd.read_csv("/mnt/HC_Volume_18315164/home-jupyter/jupyter-b-kugotov/Папка с данными для проектов/olist_orders_dataset.csv", parse_dates = True)
items_df = pd.read_csv("/mnt/HC_Volume_18315164/home-jupyter/jupyter-b-kugotov/Папка с данными для проектов/olist_order_items_dataset.csv")


# In[ ]:





# In[4]:


customer_df.head()


# In[7]:


orders_df.head()


# In[8]:


items_df.head()


# In[ ]:





# In[ ]:





# In[3]:


#Привели время к нормальому виду
orders_df["order_purchase_timestamp"] = pd.to_datetime(orders_df["order_purchase_timestamp"])
orders_df["order_approved_at"] = pd.to_datetime(orders_df["order_approved_at"])
orders_df["order_delivered_carrier_date"] = pd.to_datetime(orders_df["order_delivered_carrier_date"])
orders_df["order_delivered_customer_date"] = pd.to_datetime(orders_df["order_delivered_customer_date"])
orders_df["order_estimated_delivery_date"] = pd.to_datetime(orders_df["order_estimated_delivery_date"])

items_df["shipping_limit_date"] = pd.to_datetime(items_df["shipping_limit_date"])


# In[ ]:





# In[ ]:





# №1. Сколько у нас пользователей, которые совершили покупку только один раз? (7 баллов)

# In[15]:


orders_df["order_status"].value_counts()


# In[7]:


#Так как надо исказать пользователей, которые точно совершили покупку,
#то по моему мнению стоит оставить shipped и delivered, так как в этих случаях наш продукт точно оплатили

buy_df = orders_df.query("order_status == 'delivered' or order_status == 'shipped'")
buy_df.head()


# In[ ]:





# In[29]:


#Теперь объединяем нужные таблицы
buy_uniq_id = pd.merge(buy_df, customer_df, how = 'left', on = 'customer_id')
buy_uniq_id.shape()


# In[ ]:





# In[33]:


buy_uniq_id     .groupby('customer_unique_id', as_index = False)     .count()     .query('order_id == 1')     .shape


# Ответ: Клинтов ровно с одной покупкой 91538

# In[ ]:





# №2. Сколько заказов в месяц в среднем не доставляется по разным причинам (вывести детализацию по причинам)?

# In[36]:


#Если брать инофрмацию из информации по заказу orders_df,
#то по моему мнению надо брать типы заказов canceled
#unavailable не берем, так как информации по ним нет, может быть они были доставлены

causes_df = orders_df.query("order_status == 'canceled'")
causes_df.head()


# In[38]:


#Посчитаем сколько всего было не доставлено 
causes_df.agg({'order_id' : 'count'})


# In[ ]:





# In[47]:


#Так как в данных у нас не один год, то 
causes_df.order_estimated_delivery_date.value_counts()


# In[48]:


causes_df['month'] = causes_df["order_estimated_delivery_date"].apply(lambda x: x.strftime('%Y-%m'))
causes_df.head()


# In[ ]:





# In[ ]:





# In[57]:


causes_df_month = causes_df     .groupby("month", as_index = False)     .agg({"customer_id" : "count"})     .rename(columns = {'customer_id' : 'customer_id_count'})

causes_df_month.head()


# In[62]:


#Выведем рисунок по месяцам 

fig = plt.figure(figsize=(24,14))

sns.lineplot(data = causes_df_month, x = 'month', y = 'customer_id_count')


# In[ ]:





# In[63]:


#Теперь посчитаем среднее по месяцам
causes_df_month.agg({'customer_id_count' : 'mean'})


# Ответ в среднем в месяц не доставляется 24.038462

# In[ ]:





# №3. По каждому товару определить, в какой день недели товар чаще всего покупается. 

# In[8]:


#Для выполнения данной задачи надо взять данные из таблицы items (product_id) и объеденить их с данными из таблицы с данными о покупках.

orders = pd.merge(items_df, orders_df, how = "left", on = "order_id")
orders.head()


# In[11]:


#Далее нужно определить день недели покупки каждого заказа 
orders["Day"] = orders["order_purchase_timestamp"].dt.day_name()
orders["Day"].value_counts()


# In[17]:


#Теперь групируем нашу таблицу по product_id и дню недели
items_day_count = orders     .groupby(["product_id", "Day"], as_index = False)     .agg({"order_id" : "count"})     .rename(columns = {"order_id" : "count_per_week"})
items_day_count


# In[19]:


#Выводим самый популярный день для покупки и количество самих покупок
items_day_count.groupby(["product_id"]).max().reset_index()


# In[ ]:





# №4. Сколько у каждого из пользователей в среднем покупок в неделю (по месяцам)? Не стоит забывать, что внутри месяца может быть не целое количество недель. Например, в ноябре 2021 года 4,28 недели. И внутри метрики это нужно учесть. 

# In[23]:


#Берем сделанный ранее датафрейм buy_df, где мы брали информацию об уже купленных товарах, и объединяем их с данными по покупателям

avg_buy_person = pd.merge(buy_df, customer_df, how = "left", on = "customer_id")
avg_buy_person.head()


# In[29]:


avg_buy_person_count = avg_buy_person     .groupby(["order_purchase_timestamp","customer_unique_id"], as_index = False)     .agg({"customer_id" : "count"})     .rename(columns = {"customer_id" : "count_month"})


# In[30]:


#далее нужно узнать сколько дней в месяце и разделить на неделю (7 дней)
avg_buy_person_count["count_weeks"] = avg_buy_person_count["order_purchase_timestamp"].dt.daysinmonth / 7
avg_buy_person_count.head()


# In[33]:


#Теперь выводим информацию по средним покупкам в неделю по месяцам
avg_buy_person_count["avg_per_week"] = avg_buy_person_count["count_month"] / avg_buy_person_count["count_weeks"]
avg_buy_person_count[["customer_unique_id", "avg_per_week"]]


# In[ ]:





# № 5.1. Выполните когортный анализ пользователей.
# 
# № 5.2. В период с января по декабрь выявите когорту с самым высоким retention на 3-й месяц. Описание подхода можно найти тут. Для визуализации когортной таблицы рекомендуем использовать пример из 10-го урока python, раздел “Стильный урок”, степ 5. (15 баллов)

# Формируем когорты по совершению первой покупки. Когорты будем формировать по месяцам. Берем 2017 год, потому что по нему больше данных, чем по другим годам

# In[20]:


#Дату покупки буду считать по столбцу order_approved_at
cohort = pd.merge(customer_df, orders_df, how = "left", on = "customer_id")
cohort.head()


# In[24]:


cohort['year_month'] = cohort['order_approved_at'].dt.strftime('%Y-%m')
cohort.head()


# In[ ]:





# In[36]:


cohort_2017 = cohort[(cohort['order_approved_at'].dt.year==2017)]
cohort_2017.head()


# In[39]:


#Делим на когорты по первой покупке
cohort_df = cohort_2017.groupby('customer_unique_id', as_index=False)     .year_month.min()     .rename(columns = {"year_month" : "cohort_date"})
cohort_df.head()


# In[40]:


f_cohort_df = pd.merge(cohort_2017, cohort_df, how = "left", on = "customer_unique_id")
f_cohort_df.head()


# In[50]:


#Теперь группируем по дате начала когорты и дате покупки, считаем количество людей
cohort_finale = f_cohort_df     .groupby(["cohort_date", "year_month"], as_index = False)     .customer_unique_id.nunique()     .rename(columns = {"customer_unique_id" : "person_count"})

cohort_finale.head()


# In[51]:


#Считаем сколько покупали в месяц по количеству раз
count_person_month = cohort_2017.groupby("year_month", as_index = False)     .customer_unique_id.nunique()     .rename(columns = {"customer_unique_id" : "persons"})
count_person_month


# In[52]:


#теперь объединяем таблицы
cohort_analitics = pd.merge(cohort_finale, count_person_month, how = "left", on = "year_month")
cohort_analitics


# In[53]:


cohort_analitics["retention"] = cohort_analitics["person_count"] / cohort_analitics["persons"] * 100
cohort_analitics.head()


# In[64]:


cohort_ff = cohort_analitics.pivot(index = "cohort_date", columns = "year_month", values = "retention")
cohort_ff


# In[74]:


#Приводим таблицу к подобающему виду
cohort_ff     .style     .set_caption('User retention by cohort')     .background_gradient(cmap='viridis')     .highlight_null('white') 


# Ответ: Можно увидеть на графике, что самый большой показатель retention на 3 месяц у 5 мясяца

# Часто для качественного анализа аудитории использую подходы, основанные на сегментации. Используя python, построй RFM-сегментацию пользователей, чтобы качественно оценить свою аудиторию. В кластеризации можешь выбрать следующие метрики: R - время от последней покупки пользователя до текущей даты, F - суммарное количество покупок у пользователя за всё время, M - сумма покупок за всё время. Подробно опиши, как ты создавал кластеры. Для каждого RFM-сегмента построй границы метрик recency, frequency и monetary для интерпретации этих кластеров. Пример такого описания: RFM-сегмент 132 (recency=1, frequency=3, monetary=2) имеет границы метрик recency от 130 до 500 дней, frequency от 2 до 5 заказов в неделю, monetary от 1780 до 3560 рублей в неделю.
# 
# Создание RFM-метрик
# Для каждого пользователя рассчитаем три метрики:
# Recency (R) — время от последней покупки пользователя до текущей даты;
# Frequency (F) — суммарное количество покупок у пользователя за всё время;
# Monetary (M) — сумма покупок за всё время.

# In[8]:


#Для начала объеденим все таблицы
full_df = pd.merge(buy_df, customer_df, how = "left", on = "customer_id")
full_df = pd.merge(full_df, items_df, how = "left", on = "order_id")
full_df.head()


# In[10]:


#найдем есть ли в столбце price нули
full_df.query('price.isnull()')
full_df = full_df.dropna(subset=['price'])
full_df.head()


# In[ ]:





# In[11]:


#теперь нам надо для рассчета R найти сегодняшнюю дату
today_date = full_df['order_purchase_timestamp'].max()
today_date


# In[23]:


last_year = today_date - timedelta(days=365)
last_year = pd.to_datetime(last_year)


# In[28]:


#так как мы будем изучать данные за последний год, надо их почистить 
fill_rfm = full_df[full_df['order_purchase_timestamp'] >= last_year]
fill_rfm


# In[34]:


#теперь достолбец с количеством дней после завершенноей покупки
fill_rfm['last_day'] = (today_date - fill_rfm["order_purchase_timestamp"]).dt.days
fill_rfm


# In[ ]:





# In[36]:


#Теперь делаем первую часть RFM таблицы
RFM = fill_rfm     .groupby("customer_unique_id", as_index = False)     .agg({"last_day": lambda x: x.min(), "customer_id" : "count", "price" : "sum"})     .rename(columns = {"last_day" : "recency", "customer_id" : "frequency", "price" : "monetary"})

RFM


# In[ ]:





# In[44]:


#Смотрим распределение, чтобы потом построить рамки

plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
sns.histplot(RFM['recency'], bins = 30)

plt.subplot(1, 3, 2)
sns.histplot(RFM['frequency'], bins = 30)

plt.subplot(1, 3, 3)
sns.histplot(RFM['monetary'], bins = 40)


# In[ ]:





# In[46]:


# Задаем рамки используя квантили
quintiles = RFM[['recency', 'frequency', 'monetary']].quantile([.2, .4, .6, .8]).to_dict()
quintiles


# In[50]:


#Распределяем показатели по квантилям
#Чем больше Frequency и Monetry тем лучше. Чем меньше Recency тем лучше.

def r_score(x):
    if x <= quintiles['recency'][.2]:
        return 5
    elif x <= quintiles['recency'][.4]:
        return 4
    elif x <= quintiles['recency'][.6]:
        return 3
    elif x <= quintiles['recency'][.8]:
        return 2
    else:
        return 1

def fm_score(x, c):
    if x <= quintiles[c][.2]:
        return 1
    elif x <= quintiles[c][.4]:
        return 2
    elif x <= quintiles[c][.6]:
        return 3
    elif x <= quintiles[c][.8]:
        return 4
    else:
        return 5    


# In[52]:


RFM['R'] = RFM['recency'].apply(lambda x: r_score(x))
RFM['F'] = RFM['frequency'].apply(lambda x: fm_score(x, 'frequency'))
RFM['M'] = RFM['monetary'].apply(lambda x: fm_score(x, 'monetary'))
RFM


# In[54]:


RFM['RFM Score'] = RFM['R'].map(str) + RFM['F'].map(str) + RFM['M'].map(str)
RFM.head()


# Я собираюсь работать с 11 сегментами, основываясь на баллах R и F.
# 
# **Лидеры** приобретенные недавно, покупают часто и тратят больше всего
# 
# **Постоянные клиенты** совершают покупки на регулярной основе. Реагируют на рекламные акции.
# 
# **Потенциальные клиенты** лоялисты совершают покупки в последнее время со средней частотой.
# 
# **Недавние клиенты** совершали покупки в последнее время, но не часто.
# 
# **Перспективные покупатели** в последнее время, но потратили немного.
# 
# **Клиенты, которые нуждаются во внимании** по новизне, частоте и денежным показателям, превышающим средний уровень. Возможно, они купили не так давно.
# 
# **Вот-вот перестанут покупать** по новизне и частоте. Потеряют их, если не возобновят.
# 
# **Риск** приобретать часто, но давно. Нужно вернуть их! 
# 
# **Их нельзя потерять**, они часто покупали, но не возвращались в течение длительного времени.
# 
# **Последняя покупка была в режиме ожидания**, и количество заказов было небольшим. Могут быть утеряны.

# In[55]:


segt_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at risk',
    r'[1-2]5': 'can\'t loose',
    r'3[1-2]': 'about to sleep',
    r'33': 'need attention',
    r'[3-4][4-5]': 'loyal customers',
    r'41': 'promising',
    r'51': 'new customers',
    r'[4-5][2-3]': 'potential loyalists',
    r'5[4-5]': 'champions'
}

RFM['Segment'] = RFM['R'].map(str) + RFM['F'].map(str)
RFM['Segment'] = RFM['Segment'].replace(segt_map, regex=True)
RFM.head()


# In[ ]:





# In[58]:


#Визуализируем
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

for i, p in enumerate(['R', 'F']):
    parameters = {'R':'Recency', 'F':'Frequency'}
    y = RFM[p].value_counts().sort_index()
    x = y.index
    ax = axes[i]
    bars = ax.bar(x, y, color='silver')
    ax.set_frame_on(False)
    ax.tick_params(left=False, labelleft=False, bottom=False)
    ax.set_title('Distribution of {}'.format(parameters[p]),
                fontsize=14)
    for bar in bars:
        value = bar.get_height()
        if value == y.max():
            bar.set_color('firebrick')
        ax.text(bar.get_x() + bar.get_width() / 2,
                value - 5,
                '{}\n({}%)'.format(int(value), int(value * 100 / y.sum())),
               ha='center',
               va='top',
               color='w')

plt.show()

# plot the distribution of M for RF score
fig, axes = plt.subplots(nrows=5, ncols=5,
                         sharex=False, sharey=True,
                         figsize=(10, 10))

r_range = range(1, 6)
f_range = range(1, 6)
for r in r_range:
    for f in f_range:
        y = RFM[(RFM['R'] == r) & (RFM['F'] == f)]['M'].value_counts().sort_index()
        x = y.index
        ax = axes[r - 1, f - 1]
        bars = ax.bar(x, y, color='silver')
        if r == 5:
            if f == 3:
                ax.set_xlabel('{}\nF'.format(f), va='top')
            else:
                ax.set_xlabel('{}\n'.format(f), va='top')
        if f == 1:
            if r == 3:
                ax.set_ylabel('R\n{}'.format(r))
            else:
                ax.set_ylabel(r)
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False)
        ax.set_xticks(x)
        ax.set_xticklabels(x, fontsize=8)

        for bar in bars:
            value = bar.get_height()
            if value == y.max():
                bar.set_color('firebrick')
            ax.text(bar.get_x() + bar.get_width() / 2,
                    value,
                    int(value),
                    ha='center',
                    va='bottom',
                    color='k')
fig.suptitle('Distribution of M for each F and R',
             fontsize=14)
plt.tight_layout()
plt.show()


# In[59]:


segments_counts = RFM['Segment'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
              segments_counts,
              color='silver')
ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)
ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index)

for i, bar in enumerate(bars):
        value = bar.get_width()
        if segments_counts.index[i] in ['champions', 'loyal customers']:
            bar.set_color('firebrick')
        ax.text(value,
                bar.get_y() + bar.get_height()/2,
                '{:,} ({:}%)'.format(int(value),
                                   int(value*100/segments_counts.sum())),
                va='center',
                ha='left'
               )

plt.show()


# **Общий вывод во RFM оценке**.
# 
# Хочется сказать, что у нас очень большой процент покупателей занимают люди которые уже давно не совершали каких-либо действий 35%.
# 
# Если же говрить про лояльных покупателей, то в процентном соотношении их совсем мало, около 6%.
# 
# Также стоит отметить, что у нас давольно много новых покупателей и тех, кто много покупает по количесту но мало тратит денег, обоих по 17%
# 
# Есть еще покупатели, которые раньше много покупали, но по непонятным причинам давно этого не делали, их около 4%.

# In[ ]:





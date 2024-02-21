#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.




#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

kontrol_grubu = pd.read_excel("measurement_problems/datasets/ab_testing.xlsx", sheet_name='Control Group')
kontrol_grubu.head()

test_grubu = pd.read_excel("measurement_problems/datasets/ab_testing.xlsx", sheet_name='Test Group')
test_grubu.head()

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

kontrol_grubu.describe().T
test_grubu.describe().T

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

kontrol_grubu["Group"] = "Control"
test_grubu["Group"] = "Test"

df = pd.concat([kontrol_grubu, test_grubu], ignore_index=True, axis=0)
df.head()

#Gökberk'in çözümü
df1 = pd.concat([kontrol_grubu, test_grubu],axis=1)
df1.columns = ["Control_Impression","Control_Click","Control_Purchase","Control_Earning","Test_Impression", "Test_Click", "Test_Purchase", "Test_Earning"]
df1.head()

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.

# H0: M1 = M2
#H0: maximum bidding ve average bidding arasında fark yoktur.
# H1: M1 != M2
#H1: maximum bidding ve average bidding arasında fark vardır.

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz

df.groupby("Group").agg({"Purchase": "mean"})

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

#p-value < 0.05 H0 RED
#p-value > 0.05 H0 REDDEDİLEMEZ


######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

test_stat, pvalue = shapiro(df.loc[df["Group"] == "Control", "Purchase"])
print("Test stat : %.4f, P-value : %.4f" % (test_stat, pvalue))

#Çıktı: Test stat : 0.9773, P-value : 0.5891

test_stat, pvalue = shapiro(df.loc[df["Group"] == "Test", "Purchase"])
print("Test stat: %.4f, P-value : %.4f" % (test_stat, pvalue))
#çıktı: Test stat: 0.9589, P-value : 0.1541

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

test_stat, pvalue = levene(df.loc[df["Group"] == "Control", "Purchase"],
                           df.loc[df["Group"] == "Test", "Purchase"])

print("Test stat : %.4f, P-value : %.4f" % (test_stat, pvalue))
#çıktı: Test stat : 2.6393, P-value : 0.1083

test_stat, pvalue = ttest_ind(df.loc[df["Group"] == "Control", "Purchase"],
                              df.loc[df["Group"] == "Test", "Purchase"])
print("Test stat : %.4f, P-value : %.4f" % (test_stat, pvalue))

#çıktı: Test stat : -0.9416, P-value : 0.3493

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

p-value < ise 0.05'ten HO RED.
p-value < değilse 0.05 H0 REDDEDILEMEZ.
H0 Reddedilemez. Maximum bidding ve Average bidding arasında fark yoktur.

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
Bu projede satın alma değişkenlerinin ortalama değerleri arasında
anlamlı bir fark olup olmadığını gösteren H0 ve H1 hipotezleri tanımlanmıştır.
Daha sonra dağılımın normal olup olmadığını belirlemek için sırasıyla Shapiro ve Levene yöntemleri kullanılarak
normal ve varyans homojenlik dağılımı belirlendi.
Dağılımın normal olduğu anlaşılarak satın alma_kontrol ve satın alma_test ortalama değerleri arasında
fark olup olmadığını analiz etmek için parametrik hipotez testi uygulandı.
Buna göre satınalma_kontrol ve satın alma_test ortalama değerleri arasında fark olmadığı tespit edildi. (p>0,05)

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

Average Bidding'in Maximum Bidding'den istatistiksel olarak anlamlı bir şekilde
daha fazla dönüşüm getirdiğini doğrulayacak yeterli kanıt bulunamadığı için
maximum bidding stratejisini devam etme önerisinde bulunulabilir.


Başarı kriteri olarak belirlenen satın alma değişkenindeki değişimin istatistiksel olarak
anlamlı olmadığı ortaya çıktı. Bu nedenle elimizdeki verilerin ortalama değerleri
arasındaki farkı gösterecek kadar yeterli olmadığını ve daha fazla veriye ihtiyacımız olduğunu düşünüyorum.
Daha fazla veriye sahip olduğumuzda AB testini tekrar yapabiliriz.
Ancak mevcut haliyle satınalma_testinin bilimsel anlamda firma yararına bir avantajının olmadığı belirlendi.
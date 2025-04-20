# Fuzzy Mamdani Model with Python

<br><br>
<h1>Technology</h1> 

:snake: Python <br>
:snake: Fuzzy Model


<br><br>


<hr>
<h3>Giriş </h3>
<br>
Proje kapsamında, Nasa tarafından elde edilen aerodinamik ve akustik test veri seti
kullanılarak fuzzy model oluşturulması hedeflenmiştir. Amaç, oluşturulan fuzzy model ile
beraber, veri setinde yer alan her bir ölçüm değeri için birden fazla küme seti kurmak ( low,
high, very high vb. ) ve elde edilen kümelerin kullanımı ile beraber, gelen özellikler için en
doğru sonuç çıktısını elde etmektir.
Bu noktada, veri setinde yer alan ; <br>
● Frekans, <br>
● Saldırı açısı, <br>
● Akor uzunluğu,<br>
● Akış hızı ve <br>
● Yer değiştirme değerleri tek tek çekilerek her değişken için farklı sayıda kümeler
oluşturulmuştur ve simüle edilmiştir. <br><br>
Amaç doğrultusunda oluşturulan model, şu program akışını kullanır :<br>
1) Gerekli kütüphanelerin import edilmesi,<br>
2) Veri setinin içe aktarılması,<br>
3) Alınan veri setinin kolonlara ayrıştırılması,<br>
4) Elde edilen her kolon için bir boxplot çizilmesi ( bu boxplot’ lar, değişkenlere ait
kümelerin sınırlarını belirlemek için kullanılmıştır),<br>
5) Scipy-fuzzy kütüphanesi kullanılarak her değişken için küme setlerin oluşturulması,<br>
6) Hedef değişken olan sonuç değişkeninin de kümelere ayrılması ( farklı defuzzification
yöntemleri burada parametre aracılığıyla tek tek denenmiştir.),<br>
7) Fuzzy için kullanılacak kuralların tanımlanması,<br>
8) Kurulan kuralların, ctrl.ControlSystem() ile modele aktarılması,<br>
9) Modele aktarılan kuralların simüle edilmesi<br>
10) Veri setinde yer alan her satır için Fuzzy modelde bir sonuç üretilmesi<br>
11) Elde edilen sonuçlar için mae ve rmse değerlerinin elde edilmesi ve karşılaştırılması.<br><br><br><br>
Kod ve Çıktılar :<br>
● Gerekli kütüphanelerin import edilmesi,<br><br>

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import mean_absolute_error
```

● Veri setinin içe aktarılması,<br><br>

```
df =
pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-data
bases/00291/airfoil_self_noise.dat", header=None, sep='\t')

```

Veri seti, pandas kütüphanesi içinde yer alan read_csv yöntemi ile URL üzerinden
çekilmiştir.<br><br>
● Alınan veri setinin kolonlara ayrıştırılması,<br>

```
df.columns = ['frekans', 'derece', 'akor', 'hiz',
'yer_degistirme', 'sonuc']
```
<br>
Her kolon için bir isim verilmiştir. Bu sayede kodun ilerleyen bölümlerinde kullanımı
daha kolay olacaktır.<br><br>
● Elde edilen her kolon için bir boxplot çizilmesi ( bu boxplot’ lar, değişkenlere ait
kümelerin sınırlarını belirlemek için kullanılmıştır),<br>

```
sns.boxplot(df[['frekans']])
plt.show()
sns.boxplot(df[['derece']])
plt.show()
sns.boxplot(df[['akor']])
plt.show()
sns.boxplot(df[['hiz']])
plt.show()
sns.boxplot(df[['yer_degistirme']])
plt.show()
sns.boxplot(df[['sonuc']])
plt.show()

```
Burada elde edilen boxplot’ lar aşağıda verilmiştir.<br><br>
Akor sütunu boxplot çıktısı<br>

![akor_b](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/akor_b.jpg)

Derece sütunu boxplot çıktısı<br>
![derece_b](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/derece_b.jpg)

Frekans sütunu boxplot çıktısı<br>
![frekans_b](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/frekans_b.jpg)

 
Hiz sütunu boxplot çıktısı<br>
![hiz_b](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/hiz_b.jpg)


Yer değiştirme sütunu boxplot çıktısı<br>
![yer_degistirme_b](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/yer_degistirme_b.jpg)


Sonuç sütunu boxplot çıktısı<br>
![sonuc_b](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/sonuc_b.jpg)

<br>
● Scipy-fuzzy kütüphanesi kullanılarak her değişken için küme setlerin oluşturulması,<br>
Bu adımda, boxplot üzerinde gözlemlenen her değişken için birden fazla küme
oluşturulmuş ve küme sınırları bu boxplot’ lar üzerinden seçilmiştir. Örneğin
değerlerin en yoğun olduğu noktalar “medium” kümesi gibi bir küme sınırları içine
alınırken, ortalamadan çok fazla sapmış aykırı değerler “very high” şeklinde bir küme
sınırına alınmıştır. Bu işlemler sonucu elde edilen kümelere ait grafik çıktıları aşağıda
verilmiştir.<br>

Akor değişkeni için oluşturulan kümeler<br>
![akor](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/akor.jpg)

Derece değişkeni için oluşturulan kümeler<br>
![derece](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/derece.jpg)

Frekans değişkeni için oluşturulan kümeler<br>
![frekans](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/frekans.jpg)

Hiz değişkeni için oluşturulan kümeler<br>
![hiz](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/hiz.jpg)

Yer değiştirme değişkeni için oluşturulan kümeler<br>
![yer_degistirme](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/yer_degistirme.jpg)

<br>
● Hedef değişken olan sonuç değişkeninin de kümelere ayrılması,<br>
Sonuç için oluşturulan kümeler<br>

![sonuc](https://github.com/SeymaAtmaca/Fuzzy_Mamdani_Model_with_Pyhton/blob/main/img/sonuc.jpg)

<br>
● Fuzzy için kullanılacak kuralların tanımlanması,<br>
Fuzzy mantık için kullanılacak kurallar, sırası ile tanımlanmıştır. Aşağıda oluşturulan
kurallardan birkaçı verilmiştir.<br>

```
rule1 = ctrl.Rule((frekans_degiskeni['very_low'] |
hiz_degiskeni['very_low'] | akor_degiskeni['very_low'] |
yer_degistirme_degiskeni['very_low'] | derece_degiskeni['low']),
sonuc['very_low'])
rule5 = ctrl.Rule(frekans_degiskeni['low'] | hiz_degiskeni['low']
| akor_degiskeni['very_low'] |
yer_degistirme_degiskeni['very_low'] |
derece_degiskeni['medium'], sonuc['low'])
rule11 = ctrl.Rule(frekans_degiskeni['medium'] |
hiz_degiskeni['medium'] | akor_degiskeni['medium'] |
yer_degistirme_degiskeni['medium'] | derece_degiskeni['medium'],
sonuc['medium'])
```

<br><br>
● Kurulan kuralların, ctrl.ControlSystem() ile modele aktarılması işlemi aşağıda verilen
şekilde, scipy-fuzzy kütüphanesi fonksiyonları ile yapılmıştır.<br>

```
kurallar = ctrl.ControlSystem(rules = [rule1, rule2, rule3,
rule4, rule5, rule6, rule7, rule8, rule9,
rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,ru
le19,rule20,rule21,rule22,rule23,rule24,rule25])
```
<br><br>
● Modele aktarılan kuralların simüle edilmesi, yine scipy-fuzzy fonksiyonları ile
gerçekleştirilmiştir.<br>

```
kurallar_simul = ctrl.ControlSystemSimulation(kurallar)
```
<br><br>
● Defuzzification işlemlerinin yapılması,<br>

```
sonuc_degiskeni = ctrl.Consequent(np.arange(105, 145, 1),
'sonuc', defuzzify_method='centroid')
```
<br><br>
Bu adımda modelde sonuç değişkeni hedef değişken olduğundan defuzzification
yöntemi Consequent fonksiyonuna parametre olarak gönderilmiştir. Bu şekilde model
çalıştığında her bir satıra belirtilen metodun uygulanması sağlanmıştır. (Örneğin
burada yöntem olarak ‘centroid’ verilmiştir.)<br><br>
● Veri setinde yer alan her satır için Fuzzy modelde bir sonuç üretilmesi ;<br>

```
tahminler_cos = []
tahminler_wam = []
for i in range(len(df)):
#test verisi cekme
test = df.iloc[i]
#degerleri input olarak verme
kurallar_simul.input['frekans'] = test[0]
kurallar_simul.input['derece'] = test[1]
kurallar_simul.input['hiz'] = test[2]
kurallar_simul.input['akor'] = test[3]
kurallar_simul.input['yer_degistirme'] = test[4]



#hesaplama
kurallar_simul.compute()
#tahmin sonucunu tahmin dizisine aktarma
tahminler_cos.append(kurallar_simul.output['sonuc'])

```
<br><br>
Bu adımda; oluşturulan kurallara göre çekilen test verisi ile tahmini sonuç değişkeni
hesaplanır ve diziye eklenmiştir. Bu dizi daha sonra MAE ve RMSE değerlerini
hesaplamak için kullanılmıştır<br><br>
● Elde edilen sonuçlar için mae ve rmse değerlerinin elde edilmesi ve karşılaştırılması.<br>
#gercek sonuc degerlerinin alinmasi<br>

```
gercek_degerler = df.iloc[:,-1]
print(gercek_degerler[4])
print(tahminler_cos[4])
#basarim hesaplama
==================================================== 
#mae hesabi
mae = mean_absolute_error(gercek_degerler, tahminler_cos)
#rmse hesabi
rmse = np.sqrt(mae)
print("Results defuzzification = MAE : ", mae, "RMSE : ", rmse)

```
<br><br>
MAE (Mean Absolute Error) ve RMSE (Root Mean Square Error) modelin ürettiği
tahmin değişkenin gerçek değerle karşılaştırılmasında kullanılır.<br>
MAE, modelin ürettiği sonucun gerçek değerlerle olan farkını ölçer. MAE hesabında
her bir adımda üretilen tahmini sonuç ile gerçek sonuç değişkeninin farkının mutlak
değeri alınır ve bu hata değerleri toplanır. Daha sonra veri sayısına bölünür. Bu değer
ne kadar küçükse modelin ürettiği tahminler; gerçek değerlere o kadar yakındır denir.
RMSE; modelin ürettiği tahmin ile gerçek değer arasındaki farkı ölçmek için farkın
karesini alır böylece büyük hataların ağırlığının daha fazla olmasını sağlar. Yani
büyük hatalar daha belirgin olarak gösterilir.<br>
Modelin gerçek değerlere yakın bir sonuç üretip üretmediği RMSE değerine bakılarak
anlaşılabilir. Bu değer ne kadar küçükse tahminler gerçek değerlere o kadar yakındır.
RMSE değeri için her adımdaki tahmini ve gerçek sonuç değişkenlerinin farkının
kareleri alınır ve hata değerleri toplanıp veri sayısına bölünür. Karekökü alınır.<br><br>
Bu adımda; çekilen örnek test verisi için gerçek sonuç değişkeni 127.461 iken
centroid, som, bisector ve mom yöntemleri kullanılarak modelin oluşturduğu tahmini
sonuç değişkeninin çıktısı verilmiştir. Ayrıca her metod için hesaplanan MAE ve
RMSE değerleri de gösterilmiştir.<br><br><br>
● Centroid (Ağırlık Merkezi) Yöntemi<br>
Centroid yöntemi modeldeki çıktı kümesinin merkezini hesaplamada kullanılır. Bu
yöntemle çıktı kümesindeki noktaların üyelik dereceleriyle ağırlıklandırılmış
ortalaması bulunmaktadır. Centroid yöntemi bulanık çıktı kümesindeki belirgin
noktaları ve bu noktaların üyelik derecelerini vurgular denebilir. Etkili bir sonuç
üreten defuzzification yöntemidir.
Örnekte defuzzification yöntemi olarak ‘centroid’ parametresi verildiğinde elde edilen
tahmini sonuç değişkeni 123.34400406504066 olarak hesaplanmıştır.<br><br>
● SoM (Smallest of Maxima - En Yüksek Değerlerin En Küçüğü) Yöntemi<br>
Defuzzification yöntemi olarak kullanılan ‘som’ (Smallest of Maxima) metodunda
çıktı kümesindeki en yüksek değerler bulunur ve bunların en küçüğü seçilir. Bu
yöntem diğer berraklaştırma yöntemlerine göre daha az hassas sonuçlar üretir.
Örnek olarak alınan test verisinde tahmini sonuç değişkeni 105.0 olarak
hesaplanmıştır. Bu durumda hesaplanan MAE değerinin diğer yöntemlere göre daha
yüksek bir değer aldığı gözlenmiştir.<br><br>
● Bisector Yöntemi<br>
Bisector yönteminde çıktı kümesi alt ve üst küme olarak iki kümeye ayrılır ve bu
bölgelerin kesişim noktası çıktı değeri olarak alınır. Alınan test verisinde model
121.8908736489704 değerini üretmiştir.<br><br>
● MoM (Mean of Maxima - En Yüksek Değerlerin Ortalaması) Yöntemi<br>
Mean of Maxima (mom) yönteminde elde edilen çıktı kümesindeki en yüksek
değerlere sahip olan noktaların ortalaması çıktı olarak alınır. Bu yöntemle kümedeki
en belirgin değerler ya da en önemli değerler özetlenir ve ortalama bir çıktı üretilir.
Birçok durumda kullanıma uygun ve etkin sonuçlar üreten bir berraklaştırma
yöntemidir. Bu örnekte model tarafından 118.33 değeri üretilmiştir.<br><br><br>

<h3>Sonuç</h3>

İlgili veri seti, URL üzerinden çekilerek, oluşturulan Fuzzy modelde işlenmiştir.
Defuzzification yöntemleri karşılaştırma yapılabilmesi için denenmiştir. Kod
genelinde her satır için değerler alınır ve oluşturulan Fuzzy modelde compute()
fonksiyonu ile işlenir. Kullanılan defuzzification yöntemi de göz önüne alınarak MAE
ve RMSE hesabı yapılır. Sonuç olarak ‘centroid’ metodu kullanıldığında en iyi MAE
ve RMSE değerinin elde edildiği gözlenmiştir.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import mean_absolute_error


# veri setini yükle
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat", header=None, sep='\t')
df.columns = ['frekans', 'derece', 'akor', 'hiz', 'yer_degistirme', 'sonuc']



# Dagilimlari gozlemleme ===================================================
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




# Degiskenleri alma ve kumelere ayirma =====================================
# universe, degiskenlere ait uzayda alinabilecek tum degerlerin araligini ifade eder.
# trimf fonksiyonu, bulanik uyelik fonksiyonlarini tanimlamak icin kullanilir. 
    # ilk parametre, bulanik degisken evreni universe ,
    # ikinci parametre, ucgen uyelik fonksiyonunun alt siniri,
    # ucuncu parametre, ucgen uyelik fonksiyonunun orta siniri ( tepe ),
    # dorduncu parametre ise ucgen uyelik fonksiyonunun ust siniridir.







# UYELIK FONKSIYONLARI =====================================================
# Frekans degiskeni uyelik fonksiyonları
frekans_degiskeni = ctrl.Antecedent(np.arange(0, 8000, 1), 'frekans')
frekans_degiskeni['very_low'] = fuzz.trimf(frekans_degiskeni.universe, [0, 750, 2000])
frekans_degiskeni['low'] = fuzz.trimf(frekans_degiskeni.universe, [1500, 1750, 2500])
frekans_degiskeni['medium'] = fuzz.trimf(frekans_degiskeni.universe, [2000, 2750, 3500])
frekans_degiskeni['high'] = fuzz.trimf(frekans_degiskeni.universe, [3000, 3500, 4000])
frekans_degiskeni['very_high'] = fuzz.trimf(frekans_degiskeni.universe, [3500, 4000, 4500])
frekans_degiskeni['extreme'] = fuzz.trimf(frekans_degiskeni.universe, [4500, 5500, 8000])
frekans_degiskeni.view()

# hiz degiskeni uyelik fonksiyonları
hiz_degiskeni = ctrl.Antecedent(np.arange(30, 80, 1), 'hiz')
hiz_degiskeni['very_low'] = fuzz.trimf(hiz_degiskeni.universe, [0, 30, 40])
hiz_degiskeni['low'] = fuzz.trimf(hiz_degiskeni.universe, [35, 40, 45])
hiz_degiskeni['medium'] = fuzz.trimf(hiz_degiskeni.universe, [45, 50, 60])
hiz_degiskeni['high'] = fuzz.trimf(hiz_degiskeni.universe, [50, 60, 70])
hiz_degiskeni['very_high'] = fuzz.trimf(hiz_degiskeni.universe, [65, 70, 80])
hiz_degiskeni.view()


# Akor degiskeni uyelik fonksiyonları
akor_degiskeni = ctrl.Antecedent(np.arange(0, 0.3, 0.05), 'akor')
akor_degiskeni['very_low'] = fuzz.trimf(akor_degiskeni.universe, [0, 0.05, 0.07])
akor_degiskeni['low'] = fuzz.trimf(akor_degiskeni.universe, [0.05, 0.08, 0.1])
akor_degiskeni['medium'] = fuzz.trimf(akor_degiskeni.universe, [0.1, 0.15, 0.2])
akor_degiskeni['high'] = fuzz.trimf(akor_degiskeni.universe, [0.15, 0.20, 0.22])
akor_degiskeni['very_high'] = fuzz.trimf(akor_degiskeni.universe, [0.2, 0.25, 0.3])
akor_degiskeni.view()


# Derece degiskeni uyelik fonksiyonları
derece_degiskeni = ctrl.Antecedent(np.arange(0, 20, 1), 'derece')
derece_degiskeni['low'] = fuzz.trimf(derece_degiskeni.universe, [0, 5, 7])
derece_degiskeni['medium'] = fuzz.trimf(derece_degiskeni.universe, [5, 10, 15])
derece_degiskeni['high'] = fuzz.trimf(derece_degiskeni.universe, [10, 15, 20])
derece_degiskeni.view()

# yer değiştirme degiskeni uyelik fonksiyonları
yer_degistirme_degiskeni = ctrl.Antecedent(np.arange(0, 0.05, 0.01), 'yer_degistirme')
yer_degistirme_degiskeni['very_low'] = fuzz.trimf(yer_degistirme_degiskeni.universe, [0, 0.01, 0.02])
yer_degistirme_degiskeni['low'] = fuzz.trimf(yer_degistirme_degiskeni.universe, [0.01, 0.02, 0.03])
yer_degistirme_degiskeni['medium'] = fuzz.trimf(yer_degistirme_degiskeni.universe, [0.02, 0.03, 0.04])
yer_degistirme_degiskeni['high'] = fuzz.trimf(yer_degistirme_degiskeni.universe, [0.03, 0.04, 0.05])
yer_degistirme_degiskeni.view()


# sonuc kumeleri 
sonuc_degiskeni = ctrl.Consequent(np.arange(105, 145, 1), 'sonuc', defuzzify_method='centroid')
sonuc_degiskeni['very_low'] = fuzz.trimf(sonuc_degiskeni.universe, [0, 105, 110])
sonuc_degiskeni['low'] = fuzz.trimf(sonuc_degiskeni.universe, [105, 120, 125])
sonuc_degiskeni['medium'] = fuzz.trimf(sonuc_degiskeni.universe, [120, 125, 130])
sonuc_degiskeni['high'] = fuzz.trimf(sonuc_degiskeni.universe, [125, 130, 145])
sonuc_degiskeni.view()
plt.legend()
plt.show()






# Kurallari kurma ======================================================
# sonuc = very_low
rule1 = ctrl.Rule((frekans_degiskeni['very_low'] | hiz_degiskeni['very_low'] | akor_degiskeni['very_low'] | yer_degistirme_degiskeni['very_low'] | derece_degiskeni['low']), sonuc_degiskeni['very_low'])
rule2 = ctrl.Rule((frekans_degiskeni['very_low'] | hiz_degiskeni['low'] | akor_degiskeni['very_low'] | yer_degistirme_degiskeni['very_low'] | derece_degiskeni['low']), sonuc_degiskeni['very_low'])
rule3 = ctrl.Rule(frekans_degiskeni['very_low'] | hiz_degiskeni['very_low'] | akor_degiskeni['low'] | yer_degistirme_degiskeni['very_low'] | derece_degiskeni['low'], sonuc_degiskeni['low'])
rule4 = ctrl.Rule(frekans_degiskeni['very_low'] | hiz_degiskeni['very_low'] | akor_degiskeni['very_low'] | yer_degistirme_degiskeni['low'] | derece_degiskeni['low'], sonuc_degiskeni['low'])

# sonuc = low
rule5 = ctrl.Rule(frekans_degiskeni['low'] | hiz_degiskeni['low'] | akor_degiskeni['very_low'] | yer_degistirme_degiskeni['very_low'] | derece_degiskeni['medium'], sonuc_degiskeni['low'])
rule6 = ctrl.Rule(frekans_degiskeni['low'] | hiz_degiskeni['very_low'] | akor_degiskeni['low'] | yer_degistirme_degiskeni['very_low'] | derece_degiskeni['medium'], sonuc_degiskeni['low'])
rule7 = ctrl.Rule(frekans_degiskeni['low'] | hiz_degiskeni['very_low'] | akor_degiskeni['very_low'] | yer_degistirme_degiskeni['low'] | derece_degiskeni['medium'], sonuc_degiskeni['low'])
rule8 = ctrl.Rule(frekans_degiskeni['low'] | hiz_degiskeni['low'] | akor_degiskeni['very_low'] | yer_degistirme_degiskeni['low'] | derece_degiskeni['medium'], sonuc_degiskeni['low'])
rule9 = ctrl.Rule(frekans_degiskeni['low'] | hiz_degiskeni['very_low'] | akor_degiskeni['very_low'] | yer_degistirme_degiskeni['very_low'] | derece_degiskeni['medium'], sonuc_degiskeni['low'])
rule10 = ctrl.Rule(frekans_degiskeni['low'] | hiz_degiskeni['low'] | akor_degiskeni['low'] | yer_degistirme_degiskeni['very_low'] | derece_degiskeni['medium'], sonuc_degiskeni['low'])

# sonuc = medium
rule11 = ctrl.Rule(frekans_degiskeni['medium'] | hiz_degiskeni['medium'] | akor_degiskeni['medium'] | yer_degistirme_degiskeni['medium'] | derece_degiskeni['medium'], sonuc_degiskeni['medium'])
rule12 = ctrl.Rule(frekans_degiskeni['medium'] | hiz_degiskeni['high'] | akor_degiskeni['high'] | yer_degistirme_degiskeni['medium'] | derece_degiskeni['medium'], sonuc_degiskeni['medium'])
rule13 = ctrl.Rule(frekans_degiskeni['medium'] | hiz_degiskeni['high'] | akor_degiskeni['medium'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['medium'], sonuc_degiskeni['medium'])
rule14 = ctrl.Rule(frekans_degiskeni['high'] | hiz_degiskeni['medium'] | akor_degiskeni['high'] | yer_degistirme_degiskeni['medium'] | derece_degiskeni['high'], sonuc_degiskeni['medium'])
rule15 = ctrl.Rule(frekans_degiskeni['medium'] | hiz_degiskeni['medium'] | akor_degiskeni['medium'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['medium'], sonuc_degiskeni['medium'])
rule16 = ctrl.Rule(frekans_degiskeni['high'] | hiz_degiskeni['high'] | akor_degiskeni['medium'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['medium'], sonuc_degiskeni['medium'])
rule17 = ctrl.Rule(frekans_degiskeni['high'] | hiz_degiskeni['high'] | akor_degiskeni['high'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['medium'], sonuc_degiskeni['medium'])
rule18 = ctrl.Rule(frekans_degiskeni['very_high'] | hiz_degiskeni['high'] | akor_degiskeni['high'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['medium'], sonuc_degiskeni['medium'])

# sonuc = high
rule19 = ctrl.Rule(frekans_degiskeni['very_high'] | hiz_degiskeni['very_high'] | akor_degiskeni['very_high'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['medium'], sonuc_degiskeni['high'])
rule20 = ctrl.Rule(frekans_degiskeni['very_high'] | hiz_degiskeni['very_high'] | akor_degiskeni['high'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['medium'], sonuc_degiskeni['high'])
rule21 = ctrl.Rule(frekans_degiskeni['very_high'] | hiz_degiskeni['high'] | akor_degiskeni['very_high'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['medium'], sonuc_degiskeni['high'])
rule22 = ctrl.Rule(frekans_degiskeni['very_high'] | hiz_degiskeni['very_high'] | akor_degiskeni['very_high'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['high'], sonuc_degiskeni['high'])
rule23 = ctrl.Rule(frekans_degiskeni['very_high'] | hiz_degiskeni['very_high'] | akor_degiskeni['high'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['high'], sonuc_degiskeni['high'])
rule24 = ctrl.Rule(frekans_degiskeni['very_high'] | hiz_degiskeni['high'] | akor_degiskeni['high'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['high'], sonuc_degiskeni['high'])
rule25 = ctrl.Rule(frekans_degiskeni['very_high'] | hiz_degiskeni['very_high'] | akor_degiskeni['medium'] | yer_degistirme_degiskeni['high'] | derece_degiskeni['high'], sonuc_degiskeni['high'])


# ornek bir kuralin görüntülenmesi
rule1.view()
plt.show()

# kurallarin sistemize edilmesi
kurallar = ctrl.ControlSystem(rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule19,rule20,rule21,rule22,rule23,rule24,rule25])
# kurallarin simule edilmesi
kurallar_simul = ctrl.ControlSystemSimulation(kurallar)



# degerler uzerinde kurallar ile tahmin yapma =============================
tahminler_cos = []
tahminler_wam = []
for i in range(len(df)):
    # test verisi cekme
    test = df.iloc[i]

    # degerleri input olarak verme
    kurallar_simul.input['frekans'] = test[0]
    kurallar_simul.input['derece'] = test[1]
    kurallar_simul.input['hiz'] = test[2]
    kurallar_simul.input['akor'] = test[3]
    kurallar_simul.input['yer_degistirme'] = test[4]

    # hesaplama
    kurallar_simul.compute()


    # ================================ Defuzz kısmı ============================================
    # tahmin sonucunu tahmin dizisine aktarma
    tahminler_cos.append(kurallar_simul.output['sonuc'])
    # ==========================================================================================


# gercek sonuc degerlerinin alinamsi
gercek_degerler = df.iloc[:,-1]

print(gercek_degerler[4])
print(tahminler_cos[4])



# basarim hesaplama ====================================================
# mae hesabi
mae = mean_absolute_error(gercek_degerler, tahminler_cos)

#rmse hesabi
rmse = np.sqrt(mae)
print("Results defuzzification = MAE : ", mae, "RMSE : ", rmse)

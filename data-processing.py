'''
This script shows functions that were used for preprocessing data that was pulled from the government website.
As dataset consisted of multiple json files (one for each year) - an exemplary doc string below shows the implementation of these functions for one portion of the dataset.
'''
import pandas as pd
import math

#converts coordinates from LKS94 to WGS84. source:https://github.com/vilnius/lks2wgs/tree/master. used chatgpt 3.5 to rewrite it in python.
def grid2geo(x, y):
    pi = math.pi
    distsize = 3

    j = 0
    units = 1

    k = 0.9998
    a = 6378137
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a * a - b * b) / (a * a)
    e = math.sqrt(e2)
    ei2 = (a * a - b * b) / (b * b)
    ei = math.sqrt(ei2)
    n = (a - b) / (a + b)
    G = a * (1 - n) * (1 - n * n) * (1 + (9 / 4) * n * n + (255 / 64) * math.pow(n, 4)) * (pi / 180)
    north = (y - 0) * units
    east = (x - 500000) * units
    m = north / k
    sigma = (m * pi) / (180 * G)

    footlat = sigma + ((3 * n / 2) - (27 * math.pow(n, 3) / 32)) * math.sin(2 * sigma) + (
                (21 * n * n / 16) - (55 * math.pow(n, 4) / 32)) * math.sin(4 * sigma) + (
                          151 * math.pow(n, 3) / 96) * math.sin(6 * sigma) + (1097 * math.pow(n, 4) / 512) * math.sin(
        8 * sigma)
    rho = a * (1 - e2) / math.pow(1 - (e2 * math.sin(footlat) * math.sin(footlat)), (3 / 2))
    nu = a / math.sqrt(1 - (e2 * math.sin(footlat) * math.sin(footlat)))
    psi = nu / rho
    t = math.tan(footlat)
    x = east / (k * nu)
    laterm1 = (t / (k * rho)) * (east * x / 2)
    laterm2 = (t / (k * rho)) * (east * math.pow(x, 3) / 24) * (-4 * psi * psi + 9 * psi * (1 - t * t) + 12 * t * t)
    laterm3 = (t / (k * rho)) * (east * math.pow(x, 5) / 720) * (
                8 * math.pow(psi, 4) * (11 - 24 * t * t) - 12 * math.pow(psi, 3) * (21 - 71 * t * t) + 15 * psi * psi * (
                    15 - 98 * t * t + 15 * math.pow(t, 4)) + 180 * psi * (5 * t * t - 3 * math.pow(t, 4)) + 360 * math.pow(
                    t, 4))
    laterm4 = (t / (k * rho)) * (east * math.pow(x, 7) / 40320) * (1385 + 3633 * t * t + 4095 * math.pow(t, 4) + 1575 * math.pow(t, 6))
    latrad = footlat - laterm1 + laterm2 - laterm3 + laterm4
    lat_deg = math.degrees(latrad)

    seclat = 1 / math.cos(footlat)
    loterm1 = x * seclat
    loterm2 = (math.pow(x, 3) / 6) * seclat * (psi + 2 * t * t)
    loterm3 = (math.pow(x, 5) / 120) * seclat * (
                -4 * math.pow(psi, 3) * (1 - 6 * t * t) + psi * psi * (9 - 68 * t * t) + 72 * psi * t * t + 24 * math.pow(
            t, 4))
    loterm4 = (math.pow(x, 7) / 5040) * seclat * (61 + 662 * t * t + 1320 * math.pow(t, 4) + 720 * math.pow(t, 6))
    w = loterm1 - loterm2 + loterm3 - loterm4
    longrad = math.radians(24) + w
    lon_deg = math.degrees(longrad)

    return [lat_deg, lon_deg]


#gets desired accident info and returns a pandas dataframe.
def get_accident_info(df):

  accident_df = df.drop(df.columns[[2, 3, 4, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 35, 37, 38, 40,
                                    41, 42, 43, 44, 45, 46, 47, 50, 51]], axis=1)

  for row in range(accident_df.shape[0]):
    wgs = grid2geo(accident_df.iloc[row]["platuma"], accident_df.iloc[row]["ilguma"])
    accident_df.iloc[row, accident_df.columns.get_loc("platuma")] = wgs[0]
    accident_df.iloc[row, accident_df.columns.get_loc("ilguma")] = wgs[1]

  return accident_df



#gets desired participant info from a nested JSON and returns a pandas dataframe.
def get_participant_info(df):

  attribute_list = ["dalyvisId" , "kategorija", "lytis", "pilietybe", "amzius", "saugosDirzas", "vairuotojoKvalifikacija", "busena", "kaltininkas", "tpId"]
  column_list = ["registrokodas", "dalyvisId" , "kategorija", "lytis", "pilietybe", "amzius", "saugosDirzas", "vairuotojoKvalifikacija", "busena", "kaltininkas", "tpId"]
  participant_df = pd.DataFrame(columns=column_list)

  for row in range(df.shape[0]):
    for participant in range(len(df.iloc[row]["eismoDalyviai"])):
      new_row = [df.iloc[row]["registrokodas"]]
      for column in attribute_list:
        new_row.append(df.iloc[row]["eismoDalyviai"][participant].get(column))
      new_row = pd.Series(new_row, index=column_list)
      participant_df = pd.concat([participant_df, new_row.set_axis(participant_df.columns).to_frame().T], ignore_index=True)

  return participant_df

#gets desired automobile info from a nested JSON and returns a pandas dataframe.
def get_auto_info(df):

  attribute_list = ["tpId", "regValstybe", "kategorija", "marke", "modelis", "pagaminimoMetai", "apdraustasCivilines", "apdraustasKasko"]
  column_list = ["registrokodas", "tpId", "regValstybe", "kategorija", "marke", "modelis", "pagaminimoMetai", "apdraustasCivilines", "apdraustasKasko"]
  auto_df = pd.DataFrame(columns=column_list)

  for row in range(df.shape[0]):
    for auto in range(len(df.iloc[row]["eismoTranspPreimone"])):
      new_row = [df.iloc[row]["registrokodas"]]
      for column in attribute_list:
        new_row.append(df.iloc[row]["eismoTranspPreimone"][auto].get(column))
      new_row = pd.Series(new_row, index=column_list)
      auto_df = pd.concat([auto_df, new_row.set_axis(auto_df.columns).to_frame().T], ignore_index=True)

  return auto_df

'''
Example of 2014 data part json clean up

df_14 = pd.read_json("/content/drive/MyDrive/E_2014_12_31.json")

#dropping rows with no injuries or fatalities
df_14 = df_14.drop(df_14[(df_14.suzeistuSkaicius == 0) & (df_14.zuvusiuSkaicius == 0)].index)

#dropping rows with no coordinate info
df_14 = df_14[df_14["platuma"].notna()]
df_14 = df_14[df_14["ilguma"].notna()]

accident_df_14 = get_accident_info(df_14)
participant_df_14 = get_participant_info(df_14)
auto_df_14 = get_auto_info(df_14)

accident_df_14.to_csv('accident_14_info.csv', index=False)
participant_df_14.to_csv('participant_14_info.csv', index=False)
auto_df_14.to_csv('auto_14_info.csv', index=False)
'''

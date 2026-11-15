from mibici.modeling.version_vigente import decidir_version

CHAMPION = {"recall": 0.90, "falsas_alarmas": 1000}


def test_challenger_gana_si_mejora_el_recall():
    challenger = {"recall": 0.92, "falsas_alarmas": 1040}
    version, _ = decidir_version(CHAMPION, challenger)
    assert version == "challenger"


def test_champion_si_mejora_el_recall_pero_suben_las_falsas_alarmas():
    challenger = {"recall": 0.92, "falsas_alarmas": 1100}
    version, _ = decidir_version(CHAMPION, challenger)
    assert version == "champion"


def test_challenger_gana_si_empata_en_recall_y_baja_las_falsas_alarmas():
    challenger = {"recall": 0.905, "falsas_alarmas": 940}
    version, _ = decidir_version(CHAMPION, challenger)
    assert version == "challenger"


def test_champion_si_la_diferencia_es_menor_que_la_minima():
    challenger = {"recall": 0.9072, "falsas_alarmas": 984}
    version, _ = decidir_version({"recall": 0.9093, "falsas_alarmas": 1000}, challenger)
    assert version == "champion"


def test_champion_si_el_challenger_no_llega_al_recall_minimo():
    challenger = {"recall": 0.80, "falsas_alarmas": 500}
    version, _ = decidir_version(CHAMPION, challenger)
    assert version == "champion"


def test_champion_si_el_recall_baja_aunque_bajen_las_falsas_alarmas():
    challenger = {"recall": 0.895, "falsas_alarmas": 940}
    version, _ = decidir_version(CHAMPION, challenger)
    assert version == "champion"


def test_challenger_gana_en_la_frontera_de_recall_de_0_01():
    challenger = {"recall": 0.91, "falsas_alarmas": 1050}
    version, _ = decidir_version(CHAMPION, challenger)
    assert version == "challenger"


def test_challenger_gana_en_la_frontera_de_falsas_alarmas_del_5_por_ciento():
    challenger = {"recall": 0.90, "falsas_alarmas": 950}
    version, _ = decidir_version(CHAMPION, challenger)
    assert version == "challenger"


def test_champion_si_las_falsas_alarmas_bajan_menos_del_5_por_ciento():
    challenger = {"recall": 0.90, "falsas_alarmas": 951}
    version, _ = decidir_version(CHAMPION, challenger)
    assert version == "champion"

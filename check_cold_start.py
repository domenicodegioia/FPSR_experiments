def verifica_item_cold_start(file_training, file_test, separatore='\t', colonna_item=1):
    """
    Funzione per verificare la presenza di item nel test set
    che non esistono nel training set (scenario "cold-start").

    Args:
        file_training (str): Percorso al file di training (es. 'train.tsv').
        file_test (str): Percorso al file di test (es. 'test.tsv').
        separatore (str): Il carattere che separa le colonne (di solito tabulazione '\t').
        colonna_item (int): L'indice della colonna contenente l'ID dell'item (0 per la prima, 1 per la seconda, etc.).
    """
    try:
        # 1. Estrai tutti gli item unici dal training set
        item_nel_training = set()
        with open(file_training, 'r') as f:
            for riga in f:
                parti = riga.strip().split(separatore)
                if len(parti) > colonna_item:
                    item_nel_training.add(parti[colonna_item])

        # 2. Estrai tutti gli item unici dal test set
        item_nel_test = set()
        with open(file_test, 'r') as f:
            for riga in f:
                parti = riga.strip().split(separatore)
                if len(parti) > colonna_item:
                    item_nel_test.add(parti[colonna_item])

        # 3. Confronta i due insiemi
        item_solo_nel_test = item_nel_test - item_nel_training

        # 4. Stampa i risultati
        print("-" * 50)
        print("Analisi Item Cold-Start")
        print("-" * 50)
        print(f"Numero di item unici nel Training Set: {len(item_nel_training)}")
        print(f"Numero di item unici nel Test Set:    {len(item_nel_test)}")
        print("-" * 50)

        if item_solo_nel_test:
            print(f"RISULTATO: CONFERMATO!")
            print(f"Ci sono {len(item_solo_nel_test)} item nel test set che NON sono presenti nel training set.")
            # Se vuoi vedere alcuni esempi, decommenta la riga seguente
            # print(f"Esempi di item 'cold-start': {list(item_solo_nel_test)[:10]}")
        else:
            print("RISULTATO: NEGATIVO.")
            print("Tutti gli item presenti nel test set esistono anche nel training set.")
        print("-" * 50)

    except FileNotFoundError as e:
        print(f"ERRORE: File non trovato. Controlla il percorso: {e.filename}")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")


# --- ISTRUZIONI ---
# 1. Sostituisci con i percorsi corretti ai tuoi file.
# 2. Assicurati che 'colonna_item' sia corretta (di solito è 1 se il formato è utente-item-...).
percorso_training_set = 'data/ml-1m/train_gde.tsv'
percorso_test_set = 'data/ml-1m/test_gde.tsv'

# Esegui la verifica
verifica_item_cold_start(percorso_training_set, percorso_test_set, colonna_item=1)
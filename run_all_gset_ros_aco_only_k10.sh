#!/bin/bash

# Script per eseguire ros_aco_only con k=10 su tutti i grafi Gset
# Salva risultati finali, risultati ROS e tempi di esecuzione

# Crea la directory per i risultati se non esiste
RESULTS_DIR="results_gset_ros_aco_only_k10"
mkdir -p "$RESULTS_DIR"

# File di log principale
LOG_FILE="$RESULTS_DIR/execution_log.txt"
SUMMARY_FILE="$RESULTS_DIR/summary.csv"

# Inizializza il file di summary con l'header
echo "Gset,FinalResult,ROSResult,ExecutionTime(s),Status" > "$SUMMARY_FILE"

# Array dei numeri Gset disponibili
GSET_NUMBERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48)

# Colori per output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================" | tee -a "$LOG_FILE"
echo "Inizio esecuzione batch Gset - ros_aco_only k=10" | tee -a "$LOG_FILE"
echo "Data: $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Contatori
TOTAL=${#GSET_NUMBERS[@]}
SUCCESS=0
FAILED=0
CURRENT=0

# Loop su tutti i grafi Gset
for GSET_NUM in "${GSET_NUMBERS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    echo -e "${YELLOW}[${CURRENT}/${TOTAL}] Processando G${GSET_NUM}...${NC}" | tee -a "$LOG_FILE"
    
    # File di output per questo specifico grafo
    OUTPUT_FILE="$RESULTS_DIR/G${GSET_NUM}_output.txt"
    
    # Misura il tempo di esecuzione
    START_TIME=$(date +%s)
    
    # Esegui il comando e cattura l'output
    (cd ROS && python main.py --alg ros_aco_only --graph_type gset --gset "$GSET_NUM" --weight_mode 1 --k 10) > "$OUTPUT_FILE" 2>&1
    EXIT_CODE=$?
    
    END_TIME=$(date +%s)
    EXECUTION_TIME=$((END_TIME - START_TIME))
    
    # Verifica se l'esecuzione è andata a buon fine
    if [ $EXIT_CODE -eq 0 ]; then
        # Estrai i risultati dall'output
        FINAL_RESULT=$(grep -i "final\|result\|cut" "$OUTPUT_FILE" | tail -1 | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
        ROS_RESULT=$(grep -i "ros\|relaxed" "$OUTPUT_FILE" | grep -oE '[0-9]+(\.[0-9]+)?' | head -1)
        
        if [ -z "$FINAL_RESULT" ]; then
            FINAL_RESULT="N/A"
        fi
        if [ -z "$ROS_RESULT" ]; then
            ROS_RESULT="N/A"
        fi
        
        echo -e "${GREEN}✓ Completato - Tempo: ${EXECUTION_TIME}s - Risultato: ${FINAL_RESULT}${NC}" | tee -a "$LOG_FILE"
        echo "G${GSET_NUM},${FINAL_RESULT},${ROS_RESULT},${EXECUTION_TIME},SUCCESS" >> "$SUMMARY_FILE"
        SUCCESS=$((SUCCESS + 1))
    else
        echo -e "${RED}✗ Errore durante l'esecuzione${NC}" | tee -a "$LOG_FILE"
        echo "G${GSET_NUM},ERROR,ERROR,${EXECUTION_TIME},FAILED" >> "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
    fi
    
    echo "" | tee -a "$LOG_FILE"
done

# Statistiche finali
echo "========================================" | tee -a "$LOG_FILE"
echo "Esecuzione completata" | tee -a "$LOG_FILE"
echo "Data: $(date)" | tee -a "$LOG_FILE"
echo "Totale grafi: $TOTAL" | tee -a "$LOG_FILE"
echo -e "${GREEN}Successi: $SUCCESS${NC}" | tee -a "$LOG_FILE"
echo -e "${RED}Falliti: $FAILED${NC}" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Risultati salvati in: $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "File di summary: $SUMMARY_FILE" | tee -a "$LOG_FILE"

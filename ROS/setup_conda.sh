#!/bin/bash
# Setup script per ambiente Conda - ROS Max-k-Cut

echo "======================================"
echo "Setup Conda Environment per ROS Max-k-Cut"
echo "======================================"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "❌ Conda non è installato!"
    echo "Installa Anaconda o Miniconda da: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda trovato: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "ros-maxcut"; then
    echo "⚠️  L'ambiente 'ros-maxcut' esiste già."
    read -p "Vuoi rimuoverlo e ricrearlo? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Rimuovo l'ambiente esistente..."
        conda env remove -n ros-maxcut -y
    else
        echo "Installazione annullata."
        exit 0
    fi
fi

echo ""
echo "Creazione ambiente conda 'ros-maxcut'..."
conda env create -f environment.yml

echo ""
echo "======================================"
echo "✓ Installazione completata!"
echo "======================================"
echo ""
echo "Per attivare l'ambiente, esegui:"
echo "  conda activate ros-maxcut"
echo ""
echo "Per testare l'installazione:"
echo "  conda activate ros-maxcut"
echo "  python main.py"
echo ""
echo "Per disattivare l'ambiente:"
echo "  conda deactivate"

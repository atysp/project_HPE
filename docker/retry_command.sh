MAX_RETRIES=20  # Nombre maximal de tentatives
DELAY=5        # Délai (en secondes) entre chaque tentative

# Fonction pour exécuter la commande et vérifier le code de sortie
run_command() {
    docker build -t projet_hpe:latest .  # Remplacez "your_command_here" par la commande que vous souhaitez exécuter
}

attempt=1

while [ $attempt -le $MAX_RETRIES ]; do
    echo "Tentative $attempt..."8
    run_command
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Commande réussie."
        break
    fi
    echo "Commande échouée avec le code de sortie : $exit_code"
    if [ $attempt -lt $MAX_RETRIES ]; then
        echo "Nouvelle tentative dans $DELAY secondes..."
        sleep $DELAY
    else
        echo "Nombre maximal de tentatives atteint. Arrêt du script."
        exit 1
    fi
    attempt=$((attempt + 1))
done
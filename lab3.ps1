# Define variables
$classpath = "./lib/*;."
$mainClass = "edu.cwru.sepia.Main2"
$xmlFile = "data/labs/infexf/TwoUnitSmallMaze.xml"
$targetMessage = "The enemy was destroyed, you win!"
$runs = 150
$winCount = 0

# Run the program multiple times
for ($i = 1; $i -le $runs; $i++) {
    Write-Host "Running program $i/$runs..."
    $output = & java -cp $classpath $mainClass $xmlFile
    if ($output -like "*$targetMessage*") {
        $winCount++
    }
}

# Calculate win rate
$winRate = $winCount / $runs

# Print results
Write-Host "Total runs: $runs"
Write-Host "Wins: $winCount"
Write-Host "Win rate: $winRate"

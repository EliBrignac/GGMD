import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

# Example molecule (benzene)
mol = Chem.MolFromSmiles('c1ccccc1')

# Create a plot for the molecule
fig, ax = plt.subplots()
img = Draw.MolToImage(mol)

# Plot the molecule image
ax.imshow(img)
ax.axis('off')  # Turn off the axis

# Add the main part of the title (larger text)
plt.title("Benzene", fontsize=14)

# Add the smaller text just below the title
plt.text(0.5, 1.02, "Molecule", fontsize=8, ha='center', va='bottom', transform=ax.transAxes)

plt.show()

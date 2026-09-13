import matplotlib.pyplot as plt

# Training loss values from your actual run
losses = [0.4393, 0.3087, 0.2828, 0.2712, 0.2514, 0.2485, 0.2295, 0.1839, 0.1757, 0.1724]

plt.figure(figsize=(8,5))
plt.plot(losses, marker="o")
plt.title("Training Loss During Fine-Tuning")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.grid(True)

plt.savefig("outputs/training_loss.png")
plt.show()

print("Graph saved to outputs/training_loss.png")
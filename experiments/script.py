situation = load_situations(1)[0]

print("SITUATION:")
print(situation)

dialogue = generate_dialogue(situation)

print("\nDIALOGUE:")
for turn in dialogue:
    print("-", turn)

trajectory = extract_trajectory(dialogue, classifier)
print("\nTRAJECTORY:")
for step in trajectory:
    print(step)

print("\nDRIFT:", compute_drift(dialogue, classifier))
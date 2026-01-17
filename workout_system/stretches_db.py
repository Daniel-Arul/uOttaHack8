"""Stretch database and goal mappings"""

stretches = [
    {
        "name": "Arm Crossover Stretch",
        "target": "Shoulders",
        "duration": 30,
        "instructions": "Sit upright in chair. Bring one arm across your body at shoulder height.\n  Use opposite arm to gently pull it closer to your chest.\n  Keep shoulders relaxed. Hold and breathe.\n  Alternate sides.",
    },
    {
        "name": "Wrist Rotations",
        "target": "Wrists/Forearms",
        "reps": 20,
        "instructions": "Sit upright. Extend arms forward with palms down.\n  Rotate wrists in circles - 10 rotations clockwise, 10 counter-clockwise.\n  Keep movements smooth and controlled.\n  Repeat on both hands.",
    },
    {
        "name": "Reach Behind the Back",
        "target": "Chest/Shoulders",
        "duration": 30,
        "instructions": "Sit upright in chair. Clasp hands behind your back at lower back level.\n  Straighten arms and gently push chest forward.\n  Feel the stretch across chest and shoulders.\n  Breathe deeply and hold.",
    },
    {
        "name": "Neck Tilts",
        "target": "Neck",
        "reps": 16,
        "instructions": "Sit upright. Slowly tilt head toward one shoulder.\n  Hold for 2 seconds, then return to center.\n  Tilt toward opposite shoulder. Hold for 2 seconds.\n  Repeat alternating sides 8 times each.",
    },
    {
        "name": "Side Bends",
        "target": "Torso/Obliques",
        "reps": 16,
        "instructions": "Sit upright with feet flat. Raise one arm overhead.\n  Gently bend torso toward the opposite side.\n  Keep hips in place, bend only at the waist.\n  Alternate sides, 8 times each direction.",
    },
    {
        "name": "Chest Opener",
        "target": "Chest",
        "duration": 30,
        "instructions": "Sit upright in chair. Clasp hands behind head, elbows pointing out.\n  Gently squeeze shoulder blades together, opening chest.\n  Keep elbows back and shoulders relaxed.\n  Breathe deeply and hold.",
    },
    {
        "name": "Spinal Twist",
        "target": "Torso/Back",
        "duration": 30,
        "instructions": "Sit upright in chair. Cross arms over chest or place opposite hand to knee.\n  Rotate torso toward one side, keeping hips forward.\n  Feel the twist through entire spine.\n  Hold and breathe, then alternate sides.",
    },
    {
        "name": "Shoulder Rolls",
        "target": "Shoulders",
        "reps": 20,
        "instructions": "Sit upright with arms relaxed at sides.\n  Roll shoulders backward in smooth circles - 10 rotations.\n  Then roll forward - 10 rotations.\n  Keep movements controlled and fluid.",
    },
    {
        "name": "Neck Rolls",
        "target": "Neck",
        "reps": 12,
        "instructions": "Sit upright. Slowly drop chin to chest.\n  Roll head to one side, then toward back.\n  Continue to opposite side, then back to center.\n  Complete 6 full rotations, then reverse direction 6 times.",
    },
]

# Goal to stretches mapping
goal_to_stretches = {
    "Back Pain": ["Spinal Twist", "Side Bends", "Chest Opener"],
    "Posture": ["Chest Opener", "Reach Behind the Back", "Shoulder Rolls"],
    "Flexibility": ["Arm Crossover Stretch", "Side Bends", "Neck Rolls"],
    "Desk Break": ["Neck Tilts", "Shoulder Rolls", "Wrist Rotations"],
}

#ifndef EGG_TRAINING_H
#define EGG_TRAINING_H

// Learning rate schedule for training
inline float get_learning_rate(long step) {
    if (step < 100) return 0.5f;
    if (step < 200) return 0.25f;
    return 0.125f;
}

#endif // EGG_TRAINING_H

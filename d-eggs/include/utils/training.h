#ifndef EGG_TRAINING_H
#define EGG_TRAINING_H

// Learning rate schedule for training
inline float get_learning_rate(long step) {
    if (step < 300) return 0.1f;
    if (step < 600) return 0.05f;
    if (step < 1000) return 0.01f;
    return 0.005f;
}

#endif // EGG_TRAINING_H

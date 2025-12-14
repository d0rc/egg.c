#ifndef EGG_TRAINING_H
#define EGG_TRAINING_H

// Learning rate schedule for training
inline float get_learning_rate(long step) {
    if (step < 20) return 0.5f;
    if (step < 190) return 0.25f;
    if (step < 200) return 0.125f;
    if (step < 400) return 0.0625f;
    return 0.03f;
}

#endif // EGG_TRAINING_H

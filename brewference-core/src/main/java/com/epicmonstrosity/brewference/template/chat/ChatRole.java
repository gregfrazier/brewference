package com.epicmonstrosity.brewference.template.chat;

public enum ChatRole {
    SYSTEM("system"),
    USER("user"),
    MODEL("model");

    private final String modelName;

    ChatRole(final String modelName) {
        this.modelName = modelName;
    }

    public String getModelName() {
        return modelName;
    }
}

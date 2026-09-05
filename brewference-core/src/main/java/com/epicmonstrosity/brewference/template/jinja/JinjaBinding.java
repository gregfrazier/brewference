package com.epicmonstrosity.brewference.template.jinja;

import groovy.lang.Binding;

/** Groovy binding with Jinja-compatible reads for variables that are not defined. */
final class JinjaBinding extends Binding {
    @Override
    public Object getVariable(final String name) {
        return hasVariable(name) ? super.getVariable(name) : null;
    }
}

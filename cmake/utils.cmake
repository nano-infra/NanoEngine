
# Utility to add custom options
macro(nanodeploy_option variable description value)
    option(${variable} "${description}" ${value})
endmacro()

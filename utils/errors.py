class error(Exception):
    pass

class insufficient_data(error):
    pass

class subpop_eu(error):
    '''subpopulation definition uses event update table. Not supported.'''
    pass

class no_mapping(error):
    '''cannot find mapping for features. unable to convert back to original name.'''
    pass

class impala_error(error):
    '''impala fail to return queried results'''
    pass

class database_error(error):
    '''general database fail to return queried results'''
    pass

class config_error(error):
    '''fail to provide necessary configurations'''
    pass
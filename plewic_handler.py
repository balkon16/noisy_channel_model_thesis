import pprint
import time

pp = pprint.PrettyPrinter(indent=4)

def parse_ruby_yaml_to_python_dict(file='plewic.01.0001.yaml'):
    """
    changes Ruby-like YAML file to a list of dictionaries of dictionaries etc.

    The created dictionary has the following form:
    {
        text: ,
        new_text: ,
        attributes: {valid_sentence: },
        errors: {
            {error: , correction: , position: , attributes: {}},
            {error: , correction: , position: , attributes: {}}, ...
        }
    }
    """
    python_yaml_name = "plewic_python." + ".".join(file.split(".")[1:])

    # expressions in a line that makes the function ignore it
    # I assume that the lines containing the following strings are short enough
    # so that they don't break to another line
    # lines with other markers to be ignored may be too long for one line so
    # they are handled with last_active=3
    ignore_expr = set([":user:",
                       ":revision:",
                       "!ruby/object:"])

    # create a list that will hold dictionaries (one for each revision)
    # revision - one for every `text` key
    file_list = []

    # indicates whether the iteration process is in the errors section (
    # under `errors` marker)
    errors_section = False

    # the variable is a copy of the file given as an argument;
    # the original variable `file` can't be used because it is located in a
    # different module
    # set the variable to global so that the other function in the module
    # can use it
    global file_holder
    file_holder = file

    with open(file, 'r') as ruby_yaml:
        ignore_first_line = True

        # last active key; if -1 then no key has been active
        # active means that the key-value pair was saved to the `revision` dict
        last_active = -1

        while True:

            line = ruby_yaml.readline()

            if not line:
                break

            if ignore_first_line:
                # ignore the first line of the file that contains '---'
                ignore_first_line = False
                continue

            ignored_item_found = False

            # check if the ignored terms are in the line
            for ignored in ignore_expr:
                if ignored in line:
                    ignored_item_found = True
                    break
            if ignored_item_found:
                continue

            # go back to the line (key) that was previously active
            # text and new_text keys' values, comment values can exceed
            # character limit in a line
            active_key_lookup = {1: 'text', 2: 'new_text', 3: 'comment'}

            # errors_section is a section of the file that starts with `errors`
            # marker and ends with the last `error`'s `category` marker
            if not errors_section:

                # if `text` occured, a new revision dict must be initialised
                if '  text:' in line:
                    # if the `text` marker has appeared it means that the
                    # previous revision has been completed
                    errors_section = False
                    # time for a new revision dict; the previous one has been
                    # fully saved to the list
                    if 'revision' in locals():
                        # previous revision dict has not been saved
                        # saving it now before a new (empty) one is created
                        file_list.append(revision)

                    # no previous revision dict
                    # new empty one is created
                    revision = dict()
                    # take the second part of the split. To cover unlikely
                    # situation that `text: ` occurs more than once in a line I
                    # use [1:] instead of [1];
                    # [1:] returns a list so join() is needed
                    # rstrip() removes trailing newline character
                    revision['text'] = "".join(line.split("text:")[1:]).\
                    lstrip().rstrip()
                    # sanity check: the `text` field can't be empty
                    # if so, inform the user
                    check_if_field_is_empty(revision, "text")

                    last_active = 1
                elif '  new_text:' in line:
                    # take the second part of the split. To cover unlikely
                    # situation that `new_text: ` occurs more than once in
                    # a line I use [1:] instead of [1]; [1:] returns a list
                    # so join() is needed;
                    # rstrip() removes trailing newline character

                    revision['new_text'] = "".join(\
                    line.split("new_text:")[1:]).lstrip().rstrip()
                    # sanity check: the `new text` field can't be empty
                    # if so, inform the user
                    check_if_field_is_empty(revision, "new_text")

                    last_active = 2
                elif '    :valid_sentence:' in line:
                    # handle valid_sentence marker which is the only attribute
                    mapping_to_boolean = {'true': True, 'false': False}
                    revision['valid_sentence'] = mapping_to_boolean[\
                    line.split(":valid_sentence: ")[1].rstrip()]
                    check_if_field_is_empty(revision, "valid_sentence")
                elif '  :comment:' in line or '- :title:' in line:
                    # even though I don't save comments or titles to the
                    # dictionary I need to monitor them so that they don't
                    # get attached to other fields as defined in the `else`
                    # section of the current `if` block
                    last_active = 3

                elif '  errors:' in line:
                    # each `error` marker will be stored in a separete dict
                    # the key-value pair 'errors': [{'error': ...},
                    #                               {'error': ...}, ...]
                    # initialise an empty list for {'error': ...} dicts
                    revision['errors'] = []
                elif '    error:' in line:
                    errors_section = True
                    # start new dictionary for a single error section
                    single_error_dict = dict()
                    # take the second part of the split; I assume the
                    # second part is short enough so it doesn't exceed
                    # the line's character limit
                    single_error_dict['error'] = line.split("error: ")[1].\
                    rstrip("\n")
                    check_if_field_is_empty(single_error_dict, "error")
                elif "  attributes:\n" == line:
                    pass
                else:
                    # check if the line contains non-key i.e. text from the
                    # previous line
                    # it may be the case that the content starts with a
                    # additional whitespace (possibly whitespaces)
                    # in such a case we must only remove the first four spaces
                    # (four spaces indicate the second level of indentation)
                    # split with four spaces and take the second part

                    contents = "".join(line.split("    ")[1:]).rstrip("\n")

                    # I assume that we must add a space between the contents of
                    # the old line and the new one
                    if last_active == 1 or last_active == 2:
                        # ignore the fact that the comment (last_active=3)
                        # exceeds the character limit of the line
                        revision[active_key_lookup[last_active]] += " " + \
                        contents
            else:
                # this block starts with `correction` marker because the error
                # line has already been read
                if '    correction:' in line:
                    single_error_dict['correction'] = line.\
                    split("correction: ")[1].rstrip("\n")
                    check_if_field_is_empty(single_error_dict, "correction")
                elif '    position:' in line:
                    single_error_dict['position'] = line.\
                    split("position: ")[1].rstrip("\n")
                    check_if_field_is_empty(single_error_dict, "position")
                elif '    attributes:' in line:
                    # handle attributes of the error
                    # the `attributes` key corresponds to a dict containing
                    # attributes (indented by additional two whitespaces)
                    error_attributes_dict = dict()
                    continue
                elif "      :type:" in line:
                    # the last character in the split() argument is ':' because
                    # each category starts with ":"
                    error_attributes_dict['type'] = line.split("type: :")[1].\
                    rstrip("\n")
                    check_if_field_is_empty(error_attributes_dict, 'type')
                elif "      :distance:" in line:
                    error_attributes_dict['distance'] = line.\
                    split("distance: ")[1].rstrip("\n")
                    check_if_field_is_empty(error_attributes_dict, 'distance')
                elif "      :category:" in line:
                    # the `category` marker is the last attribute of an error
                    # in other words it concludes the single error section
                    # the single error dict must be appended to the errors list
                    errors_section = False
                    error_attributes_dict['category'] = line.\
                    split("category: ")[1].rstrip("\n")
                    check_if_field_is_empty(single_error_dict, "category")
                    # save the error_attributes_dict
                    single_error_dict['attributes'] = error_attributes_dict
                    # the single error dict must be appended to the errors list
                    revision['errors'].append(single_error_dict)

    return file_list

def check_if_field_is_empty(dict, key):
    """
    Given a dictionary and a key (string) checks whether the value in the
    `revision` dict is an empty string. If the field is empty True is returned.
    """
    try:
        if dict[key] == "":
            print("The field ", key, " is empty! Revision number is ", \
            current_revision, " in the file ", file_holder)
            return True
    except KeyError:
        pass

if __name__ == "__main__":
    result = parse_ruby_yaml_to_python_dict("plewic.07.0133.yaml")
    print(len(result))

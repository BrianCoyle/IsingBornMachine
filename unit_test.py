from file_operations_out import  string_to_int_byte
from run_and_compare import bytes_to_int, read_ints_from_file

def create_test_binary_file(file_name, int_list):

    with open(file_name, 'wb') as f:
        
        for i in int_list:

            f.write(bytes([i]))

def test_string_to_int_byte():
    assert string_to_int_byte('101', 3, 0)  == 5
    assert string_to_int_byte('1011010101011', 13, 0)  == 181
    assert string_to_int_byte('1011010101011', 13, 1)  == 11

def test_bytes_to_int():
    assert bytes_to_int([5,4,1]) == 328705

def test_read_ints_from_file():

    file_name = 'binary_file_system_example'

    create_test_binary_file(file_name, [4,6,18])

    with open(file_name, 'rb') as f:

        int_list = read_ints_from_file(5,3,f)

    assert int_list == [4,6,18]

    create_test_binary_file(file_name, [8 , 9 , 18 , 7])

    with open(file_name, 'rb') as f:

        int_list = read_ints_from_file(16, 2, f)

    assert int_list == [2057, 4615]

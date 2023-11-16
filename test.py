def check_list_dims2(lst):
    max_num = 4096
    max_product = 4096 * 4096
    if len(lst) != 2:
        return False
    num1, num2 = lst
    if num1 > max_num or num2 > max_num:
        return False
    if num1 * num2 > max_product:
        return False
    return True

def check_list_dims3(lst):
    max_num = 4096
    max_product_two = 100000
    max_product_three = 2000000
    if len(lst) != 3:
        return False
    num1, num2, num3 = lst
    if num1 > max_num or num2 > max_num or num3 > max_num:
        return False
    if num1 * num2 > max_product_two or num1 * num3 > max_product_two or num2 * num3 > max_product_two:
        return False
    if num1 * num2 * num3 > max_product_three:
        return False
    return True


def check_list_dims4(lst):
    max_num = 4096
    max_product_two = 100000
    max_product_three = 3000000
    max_product_four = 8000000
    if len(lst) != 4 or any(num > max_num for num in lst):
        return False
    products_two = [lst[i] * lst[j] for i in range(3) for j in range(i+1, 4)]
    if any(product > max_product_two for product in products_two):
        return False
    products_three = [lst[i] * lst[j] * lst[k] for i in range(2) for j in range(i+1, 3) for k in range(j+1, 4)]
    if any(product > max_product_three for product in products_three):
        return False
    product_four = lst[0] * lst[1] * lst[2] * lst[3]
    if product_four > max_product_four:
        return False
    return True







# #include <iostream>
# #include <vector>

# bool checkList(const std::vector<int>& lst) {
#     int maxNum = 4096;
#     int maxProductTwo = 100000;
#     int maxProductThree = 3000000;
#     int maxProductFour = 8000000;

#     if (lst.size() != 4) {
#         return false;
#     }

#     for (int i = 0; i < 4; i++) {
#         if (lst[i] > maxNum) {
#             return false;
#         }
#     }

#     for (int i = 0; i < 3; i++) {
#         for (int j = i + 1; j < 4; j++) {
#             if (lst[i] * lst[j] > maxProductTwo) {
#                 return false;
#             }
#         }
#     }

#     for (int i = 0; i < 2; i++) {
#         for (int j = i + 1; j < 3; j++) {
#             for (int k = j + 1; k < 4; k++) {
#                 if (lst[i] * lst[j] * lst[k] > maxProductThree) {
#                     return false;
#                 }
#             }
#         }
#     }

#     int productFour = lst[0] * lst[1] * lst[2] * lst[3];
#     if (productFour > maxProductFour) {
#         return false;
#     }

#     return true;
# }

# int main() {
#     std::vector<int> list1 = {2048, 1024, 512, 256};
#     std::cout << std::boolalpha << checkList(list1) << std::endl;  // 输出：true

#     std::vector<int> list2 = {4097, 2048, 1024, 512};
#     std::cout << std::boolalpha << checkList(list2) << std::endl;  // 输出：false

#     std::vector<int> list3 = {2048, 2048, 2048, 2048};
#     std::cout << std::boolalpha << checkList(list3) << std::endl;  // 输出：false

#     return 0;
# }

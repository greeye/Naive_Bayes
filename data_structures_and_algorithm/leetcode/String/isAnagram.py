class Solution:
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # 方法一  : 思路排序后相关即为异位词
        # return sorted(s)==sorted(t)
        # 方法二  : 思路 1.字符串长度大小  2. 一一对比字符
        if len(s) != len(t):
            return False
        if s == t == "":
            return False
        c = set(t)
        for i in c:
            if t.count(i) != s.count(i):
                return False
        return True






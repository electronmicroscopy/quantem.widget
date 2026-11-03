"""Detect private computer, person, partner, dataset and path names in public text.

Every name is stored as the SHA-256 digest of its lowercase spelling, so this
public guard does not spell out what it forbids. Public text names hardware
("MacBook Pro (M5 Max, 128 GB)") and public dataset ids, never a host nickname,
a tailnet address, a person outside CITATION.cff, an industrial partner, or a
collaborator's sample or session name.

Three kinds of match are checked:

- word names: each word, and each pair of adjacent words, of the text;
- whole-word names: a token bounded by characters other than letters, digits
  and underscores;
- substring names: any substring of a letter/digit/``_``/``-`` token, matched
  case-insensitively, because session stems embed them (``<name>_013``).
"""

import functools
import hashlib
import re

# Lowercase single words or two-word phrases.
WORD_DIGESTS = {
    "0522a55e2d5f0993a3d66d28864b2862a7218a75ea7968b075333434404485c3",
    "0ec5f7c0abe435d22cca61c6d4ed58226e83f9cecef7dcbfda3b48c449c77b35",
    "3b94d6c610c206fc29004242626f013dec81548647d03dcd52051e1ac83db41f",
    "4b25923a4f31b83195ceb0d160e0a1a6d1556b5a2cd4c582ff40aa0498ae1578",
    "57e68ee3e8de805cdff477987deac14ad30d1fe306d2abe904b580f49dcbf34a",
    "7a183d8629b0c640adb76efd475fb8749627f8b1890674b0d947107d28562075",
    "955867566f1ea1983014810667cb92d853d8e7bc3cf1a9fe0321510add699536",
    "968e2d5b08687bf42997461cbdef6c844eabbf04f440cee888c95b864c2a4bcc",
    "a1df6dffb5e4869e2f750c4121219699b9f6d748d16b7118494d1f7dcf1585e5",
    "db4b708792563afeb166543dccefd749ee4eaa0e9b64775c1a58ba0a1b1c0a6f",
    "ee4a15f46f64a7ad42edeb2b2b04f2de4d384470556d20daf62c37cc1ef46571",
    "f2f8cc1383952f00b3d4ed4989c9ef0eea6df87df4f238791cbc38a35859c9af",
    "fee4822e086bde0ca5facf96fd4a17210cc3745f0c5ef638aac0b80a6b835699",
}
WHOLE_WORD_DIGESTS = {
    "4174e7f6203eebdf2788d6ff419f4b84b53ac5480f630daf7d5f04c291098084",
    "b613030f8c7be797e7a45002505c8a79b632db50c2d401c739cd2bec47a4017f",
}
# Keyed by the length of the lowercase name.
SUBSTRING_DIGESTS = {
    4: {
        "29d21b963e339e351debd891fee6a7243dd872e54f1a02bb2e5c2dc48330d155",
        "3ac7adb763765425bafb237c5fe8b009a10f30164d219bb547d32cc9da7a6217",
        "414a698b093ef3c6e3eb11ec9cbbc8081b9490291a04d66f9e54192028693507",
    },
    5: {
        "0ec5f7c0abe435d22cca61c6d4ed58226e83f9cecef7dcbfda3b48c449c77b35",
    },
    6: {
        "05b9b503fab8dc160749f20292d59e878d2c1c111280da9f8b4fd4079043dba2",
        "11a92a1866a408344175415d5805f9a8275683d4b8f4b28d3668a790bf181fc2",
        "ee110021d2fb4d9a3a330b3b5a65ccc80743a0bfce4d469a984cd62f9472c3c7",
    },
    7: {
        "6156897ffdc062402e9f4b76ee880e4f731536b13c7846c8eb2d70aad068c565",
        "73095e0ea475a9c05368a611dc3a90507ed922987ced2b03eb1ef3d3ff6c493b",
        "968e2d5b08687bf42997461cbdef6c844eabbf04f440cee888c95b864c2a4bcc",
    },
    8: {
        "5dba284a76095ddea105c8b5b20c2abdf61efcb1e580114edf9734d3a972e710",
    },
    9: {
        "76538428d140b43b268d560184c8447326e6715847bebe544b737b1f86a78239",
    },
    10: {
        "398d6aba34957a2c183a646fc2770881c9a74c9db0b2f71f1311a2425231e0d7",
        "be1ce0706be9d55cc239b97a97b3bf5ed71478db1905e069f8c83e394292ff15",
    },
    11: {
        "f45f6c71caf6df04b6d8ec6dd6533543cf9efc2dfc5265da0c003b27fcbcaf53",
    },
    14: {
        "33ce459df5d7326181e32fd93c8ef6f2446f40f202259279a649c398ec0f9369",
    },
}
# Built from parts so the guard does not contain the paths it forbids.
PRIVATE_SUBSTRINGS = ("/home/" + "owner", "/Users/" + "macbook", "owner-" + "ms-7e34")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@functools.cache
def _token_is_private(token: str) -> bool:
    """Whether any substring of one lowercase token is a substring name."""
    for length, digests in SUBSTRING_DIGESTS.items():
        for start in range(len(token) - length + 1):
            if _digest(token[start : start + length]) in digests:
                return True
    return False


def mentions_private_name(text: str) -> bool:
    """Whether ``text`` names a private computer, person, partner, dataset or path.

    Words are split at letters only (``host2`` yields ``host``) and at letters
    and digits (a tailnet id such as ``tail1234`` stays one word); each word and
    adjacent pair is compared with ``WORD_DIGESTS``. Tokens of letters, digits
    and underscores are compared with ``WHOLE_WORD_DIGESTS``, and every
    substring of a letter/digit/``_``/``-`` token with ``SUBSTRING_DIGESTS``.
    Distinct tokens are hashed once per process, which keeps a scan of the
    whole repository fast.
    """
    if any(substring in text for substring in PRIVATE_SUBSTRINGS):
        return True
    lower = text.lower()
    for words in (re.findall(r"[a-z]+", lower), re.findall(r"[a-z0-9]+", lower)):
        candidates = {*words, *(f"{first} {second}" for first, second in zip(words, words[1:]))}
        if any(_digest(candidate) in WORD_DIGESTS for candidate in candidates):
            return True
    if any(_digest(token) in WHOLE_WORD_DIGESTS for token in set(re.findall(r"[a-z0-9_]+", lower))):
        return True
    return any(_token_is_private(token) for token in set(re.findall(r"[a-z0-9_-]+", lower)))

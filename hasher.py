import streamlit_authenticator as stauth

hashed_passwords = stauth.Hasher(['netspend24', 'netspend25']).generate()

print(hashed_passwords)
# TODO : 기상청 API 연동 로직

"""
TODO : 지점 반환 함수. - https://apihub.kma.go.kr/api/typ01/url/stn_inf.php?inf=SFC&stn=&tm=202604290900&help=1&authKey=ya4MB9p0TS-uDAfadL0vAA 여기 데이터 참고,
    요청 파라미터(stn_city : 주소(행정동 까지) ex)"경기도 수원시권선구 고색동")
    응답(stn : 지점 번호 ex) 119


TODO : 기상청 API 연동 함수.
    요청 파라미터( stn : 지점 번호 , tm1 : 현재 년월일시분 ex)202605010100 )
    응답 (
        TA :기온(°C),
        RN :강수량(mm),
        WS :풍속(m/s),
        HM :상대습도(%),
        SS :일조(hr),
        SI :일사(MJ/m2)
        ) - 값을 저 순서대로 반환.
"""
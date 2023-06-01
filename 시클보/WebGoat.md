# WebGoat 실습
## 목차
1. A1 Injection  
    1-1 [(A1) Injection(intro)](#sql-injectionintro)  
    1-2 [(A2) SQL Injection(advanced)](#sql-injectionadvanced)  
## (A1) Injection
### SQL Injection(intro)
<b>[2 개념]</b>
SQL(Structured Query Language)이란?  
: 관계형 databases를 관리하고 database 안에서 data에서 다양한 작업을 수행하기 위해 사용되는 표준화된(ANSI in 1986, ISO in 1987) 프로그래밍 언어이다.  

database : data의 모임  
data : 행들, 열들, 테이블들로 구성됨 그리고 관련 정보를 쉽게 찾을 수 있도록 indexing 됨  
<br>  

[2의 문제]  
아래 Employees Table에서 Bob Franco의 부서를 이끌어 내라  
<img src = "../image/SQL[2] 문제.png" width ="100%">

[2의 답]    
<img src = "../image/SQL[2]답.png" width ="100%">

<b>[3 개념]</b>
DML(Data Manipulation Language)란?  
: 데이터를 다룬다.  
: SELECT, INSERT, UPDATE, DELETE 등과 같은 가장 일반적인 SQL문을 포함한다.  

- SELECT : 하나의 database로부터 데이터를 도출한다.
- INSERT : 하나의 table에 데이터를 삽입한다.
- UPDATE : 하나의 table 내부에 존재하는 데이터를 update(수정) 한다.
- DELETE : 한 database table로부터 모든 기록들을 삭제한다.

### SQL Injection(advanced)
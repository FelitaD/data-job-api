from flask_restful import Resource, reqparse
from flask_jwt_extended import create_access_token, create_refresh_token, jwt_required, get_jwt_identity
from hmac import compare_digest


from models.user import UserModel

_parser = reqparse.RequestParser()
_parser.add_argument('username', type=str, required=True, help="This field cannot be blank")
_parser.add_argument('password', type=str, required=True, help="This field cannot be blank")


class UserLogin(Resource):

    def post(self):

        data = _parser.parse_args()
        user = UserModel.find_by_username(data['username'])

        if user and compare_digest(user.password, data['password']):
            access_token = create_access_token(identity=user.username, fresh=True)
            refresh_token = create_refresh_token(user.username)
            return {
                'access_token': access_token,
                'refresh_token': refresh_token
            }, 200

        return {'message': 'Invalid credentials'}, 401


class UserRegister(Resource):

    def post(self):
        data = _parser.parse_args()

        if UserModel.find_by_username(data['username']):
            return {'message': 'A user with that username already exists'}, 400

        user = UserModel(**data)
        user.save_to_db()

        return {"message": "User created succesfully."}, 201

class UserPermissions(Resource):

    @jwt_required()
    def post(self):

        user = UserModel.find_by_username(get_jwt_identity())

        return {
            'user': user.username,
            'permission v1': user.v1,
            'permission v2': user.v2
        }, 200


class UserWelcome(Resource):

    def get(self, name):
        return "Welcome {}".format(name)

